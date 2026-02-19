import torch
import os
import numpy as np
import json
import pickle
import sys
from collections import OrderedDict
import logging
import PoseTools
#import tfdatagen
import time
#import tensorflow.compat.v1 as tf   # type:ignore
# ISTFR = False
# try:
#     from tfrecord.torch.dataset import TFRecordDataset
#     ISTFR = True
# except:
#     print('TFRecordDataset not available')
#import errno
#import re
#import gc
import time
import cv2
import xtcocotools.mask
from functools import partial
import random
import time
# from torch import autograd
# autograd.set_detect_anomaly(True)

#ISWINDOWS = os.name == 'nt'

def print_train_data(cur_dict):
    p_str = ''
    for k in cur_dict.keys():
        p_str += '{:s}:{:.2f} '.format(k, cur_dict[k])
    logging.info(p_str)

def decode_augment(features, conf, distort):
    n_pts = conf.n_classes
    h = features['height'][0]
    w = features['width'][0]
    d = features['depth'][0]
    features['image_raw'] = np.array(features['image_raw'])
    ims = features['image_raw'].reshape([1,h,w,d])

    if 'mask' in features.keys():
        features['mask'] = np.array(features['mask']).reshape([h,w,1])
        if conf.multi_use_mask:
            ims = ims * features['mask']
    else:
        features['mask'] = None

    if 'max_n' in features.keys():
        n_max = features['max_n'][0]
        ntgt = features['ntgt'][0]
    else:
        n_max = None
        ntgt = None

    if 'occ' in features.keys():
        features['occ'] = features['occ'].reshape([-1,n_pts])
    elif n_max is None:
        features['occ'] = np.zeros([1, n_pts])
    else:
        features['occ'] = np.zeros([n_max, n_pts])

    if n_max is None:
        locs = features['locs'].reshape([1,n_pts,2])
        features['occ'] = features['occ'][0,...]
    else:
        locs = np.ones([1,n_max,n_pts,2])*np.nan
        locs[:,:ntgt] = features['locs'].reshape([1,ntgt,n_pts,2])

    if 'trx_ndx' not in features.keys():
        features['trx_ndx'] = np.array([0])


    features['info'] = np.array([features['expndx'][0],features['ts'][0],features['trx_ndx'][0]])


    ret = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.rescale,mask=features['mask'][None,...,0])
    ims,locs = ret[:2]
    if features['mask'] is not None:
        features['mask'] = ret[2][0]
    else:
        features['mask'] = np.array([])

    # convert CHW format
    ims = np.transpose(ims[0,...]/255.,[2,0,1])

    features['images'] = ims
    features['locs'] = locs[0,...]

    return features

def dataloader_worker_init_fn(id,epoch=0):
    np.random.seed(id + 100*epoch)#+int(time.time()))
    random.seed(id+100*epoch)#+int(time.time()))

class coco_loader(torch.utils.data.Dataset):

    def __init__(self, conf, ann_file, augment, image_folder_path=''):
        self.ann = PoseTools.json_load(ann_file)
        self.conf = conf
        self.augment = augment
        self.len = max(conf.batch_size,len(self.ann['images']))
        self.ex_wts = torch.ones(self.len)
        self.coco_ignore_mask =  conf.get('coco_ignore_mask', False)
        if self.len > 0:
            # Test the given image_folder_path.  If it's no good, use conf.coco_im_dir
            test_image_file_name = self.ann['images'][0]['file_name']
            test_image_file_path = os.path.join(image_folder_path, test_image_file_name)            
            if os.path.exists(test_image_file_path):
                self.image_folder_path = image_folder_path
            else:               
                backup_image_folder_path = conf.coco_im_dir
                logging.warning('No file at %s, trying %s as the coco_loader image_folder_path' % (test_image_file_path, backup_image_folder_path))
                test_image_file_path = os.path.join(image_folder_path, test_image_file_name)            
                if os.path.exists(test_image_file_path):
                    self.image_folder_path = backup_image_folder_path
                else:
                    raise RuntimeError('Unable to read a test image in coco_loader __init__() method')
        else:
            # If there are no images, this likely doesn't matter anyway
            self.image_folder_path = image_folder_path

    def __len__(self):
        return max(self.conf.batch_size,len(self.ann['images']))

    def pad(self,image):
        # Get the original image dimensions
        original_height, original_width = image.shape[:2]

        if self.conf.pad_images:
            padding_w = self.conf.imsz[1] - original_width
            padding_h = self.conf.imsz[0] - original_height

            padded_image = cv2.copyMakeBorder(image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT)
            return padded_image


        # Calculate the aspect ratio of the original image
        original_aspect_ratio = original_width / original_height

        # Calculate the desired width and height based on the target aspect ratio
        target_aspect_ratio = self.conf.imsz[1]/self.conf.imsz[0]
        if original_aspect_ratio < target_aspect_ratio:
            # Pad vertically
            desired_width = int(original_height * target_aspect_ratio)
            padding = (desired_width - original_width)
            padded_image = cv2.copyMakeBorder(image, 0, 0, 0, padding, cv2.BORDER_CONSTANT)
        elif original_aspect_ratio > target_aspect_ratio:
            # Pad horizontally
            desired_height = int(original_width / target_aspect_ratio)
            padding = desired_height - original_height
            padded_image = cv2.copyMakeBorder(image, 0, padding, 0, 0, cv2.BORDER_CONSTANT)
        else:
            # No padding required, aspect ratios are the same
            padded_image = image

        return padded_image

    def __getitem__(self, item):
        conf = self.conf
        if (self.conf.batch_size)> len(self.ann['images']):
            item = np.random.randint(len(self.ann['images']))
        im_path = os.path.join(self.image_folder_path, self.ann['images'][item]['file_name'])
        if not os.path.exists(im_path):
            im_path = os.path.join(conf.coco_im_dir,im_path)

        im = cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
        if im.ndim == 2:
            im = im[...,np.newaxis]
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if im.shape[2] == 1:
            im = np.tile(im,[1,1,3])

        if im.shape[:2] != tuple(conf.imsz):
            im = self.pad(im)
            sfactor = (im.shape[0]/conf.imsz[0]+im.shape[1]/conf.imsz[1])/2
        else:
            sfactor = 1.


        if 'movid' not in self.ann['images'][item] or type(self.ann['images'][item]['movid']) == list:
            info = [item,item,item]
        else:
            info = [self.ann['images'][item]['movid'], self.ann['images'][item]['frm'],self.ann['images'][item]['patch']]

        info = np.array(info)
        curl = np.ones([conf.max_n_animals,conf.n_classes,3])*-10000
        lndx = 0
        annos = []
        for a in self.ann['annotations']:
            if not (a['image_id']==self.ann['images'][item]['id']):
                continue
            locs = np.array(a['keypoints'])
            if a['num_keypoints']>0 and a['area']>1:
                locs = np.reshape(locs, [conf.n_classes, 3])
                # if np.all(locs[:,2]>0.5):
                curl[lndx,...] = locs
                lndx += 1
            annos.append(a)

        curl = np.array(curl)
        if conf.nan_as_occluded:
            occ = curl[...,2] < 0.5
        else:
            occ = curl[...,2] < 1.5
        locs = curl[...,:2]
        if np.all(locs[curl[...,2]==0,:]==0):
            locs[curl[...,2]==0,:] = np.nan

        if self.coco_ignore_mask:
            mask = self.get_mask_ignore(annos,im.shape[:2])
        else:
            mask = self.get_mask(annos,im.shape[:2])
        im,locs, mask,occ = PoseTools.preprocess_ims(im[np.newaxis,...], locs[np.newaxis,...],conf, self.augment, conf.rescale*sfactor, mask=mask[None,...],occ=occ[None])
        osz = tuple([round(ii/conf.rescale) for ii in conf.imsz][::-1])
        if (im.shape[1] != osz[1]) or (im.shape[2] != osz[0]):
            im = cv2.resize(im[0,...],osz)[None,...]
            mask = cv2.resize(mask[0,...].astype(np.float32),osz)[None,...]
        im = np.transpose(im[0,...] / 255., [2, 0, 1])
        mask = mask[0,...]
        if not self.conf.is_multi:
            locs = locs[:,0]
            occ = occ[:,0]

        features = {'images':im,
                    'locs':locs[0],
                    'info':info,
                    'occ': occ[0],
                    'mask':mask,
                    'item':item
                    }
        return features

    def get_mask(self, anno, im_sz):
        conf = self.conf
        m = np.zeros(im_sz,dtype=np.float32)

        if not conf.multi_loss_mask:
            return m<0.5

        for obj in anno:
            if 'bbox' in obj:
                # MK 20230531. Segmentation is required to be a numpy array now. Probably has to do with the newer ampere image. xtcocotools might have been updated. Sigh.
                # MK 20230823. when the input object is an np.array, xtcocotools expects it to be in bbox format [x,y,width,height]
                # rles = xtcocotools.mask.frPyObjects(
                #     obj['segmentation'], im_sz[0],
                #     im_sz[1])
                rles = xtcocotools.mask.frPyObjects(
                    # np.array(obj['segmentation']).astype('double'), im_sz[0],
                    # im_sz[1])
                    np.array([obj['bbox']]).astype('double'), im_sz[0],im_sz[1])
                for rle in rles:
                    m += xtcocotools.mask.decode(rle)
                # if obj['iscrowd']:
                #     rle = xtcocotools.mask.frPyObjects(obj['segmentation'],im_sz[0], im_sz[1])
                #     m += xtcocotools.mask.decode(rle)
                # else:
        return m>0.5
    def get_mask_ignore(self, anno, im_sz):
        conf = self.conf
        m = np.zeros(im_sz,dtype=np.float32)

        if not conf.multi_loss_mask:
            return m<0.5

        for obj in anno:
            if ('bbox' in obj) and ('ignore' in obj) and obj['ignore']:
                rles = xtcocotools.mask.frPyObjects(
                    np.array([obj['bbox']]).astype('double'), im_sz[0],im_sz[1])
                for rle in rles:
                    m -= xtcocotools.mask.decode(rle)
        m = (m+1).clip(0,1)
        for obj in anno:
            if ('bbox' in obj) and not  (('ignore' in obj) and obj['ignore']):
                rles = xtcocotools.mask.frPyObjects(
                    np.array([obj['bbox']]).astype('double'), im_sz[0],im_sz[1])
                for rle in rles:
                    m += xtcocotools.mask.decode(rle)

        return m>0.5

    def update_wts(self,idx,loss):
        for ix,l in zip(idx,loss):
            self.ex_wts[ix] = torch.tensor(l)


class PoseCommon_pytorch(object):

    def __init__(self,conf,name='deepnet',usegpu=True,zero_seeds=False,img_prefix_override=None,debug=False):
        self.conf = conf
        self.name = name
        self.prev_models = []
        self.td_fields = ['dist','loss']
        self.train_epoch = 1
        self.zero_seeds = zero_seeds
        self.img_prefix_override = img_prefix_override
        self.debug = debug
        # conf.is_multi = is_multi

        if usegpu and torch.cuda.is_available() and not conf.get('use_openvino',False):
            self.device = "cuda"
        else:
            self.device = "cpu"
            if usegpu and not conf.get('use_openvino',False):
                logging.warning("CUDA Device not available. Using CPU!")

        if conf.db_format == 'coco':
            self.use_hard_mining = conf.get('use_hard_mining', False)
        else:
            self.use_hard_mining = False

    def get_ckpt_file(self):
        return os.path.join(self.conf.cachedir,self.name + '_ckpt')

    def get_latest_model_file(self):
        ckpt_file = self.get_ckpt_file()
        if not os.path.exists(ckpt_file):
            model_file = None
        else:
            with open(ckpt_file,'r') as f:
                mf = json.load(f)
            if len(mf) > 0:
                model_file = mf[-1]
            else:
                model_file = None
            model_file = os.path.join(self.conf.cachedir,model_file)
        return  model_file

    def get_td_file(self):
        if self.name =='deepnet':
            td_name = os.path.join(self.conf.cachedir,'traindata')
        else:
            td_name = os.path.join(self.conf.cachedir, self.conf.expname + '_' + self.name + '_traindata')
        return td_name

    def save(self, step, model, opt, sched):
        fname = self.name + '-{}'.format(step)
        out_file = os.path.join(self.conf.cachedir,fname)
        torch.save({'step':step, 'model_state_params':model.state_dict(), 'optimizer_state_params':opt.state_dict(), 'sched_state_params':sched.state_dict()
                    },
                    out_file)
        logging.info('Saved model to {}'.format(out_file))
        self.save_td()
        self.prev_models.append(fname)
        if len(self.prev_models) > self.conf.maxckpt:
            for curm in self.prev_models[:-self.conf.maxckpt]:
                if os.path.exists(os.path.join(self.conf.cachedir,curm)):
                    os.remove(os.path.join(self.conf.cachedir,curm))
            _ = self.prev_models.pop(0)

        with open(self.get_ckpt_file(),'w') as cf:
            json.dump(self.prev_models,cf)

    def convert_openvino(self,model,model_file):
        from openvino.runtime import Core

        model = model.cpu()
        model.eval()

        conf = self.conf
        dummy_input = {'images': torch.randn([1, 3, int(conf.imsz[0]//conf.rescale),int(conf.imsz[1]//conf.rescale)])}
        if not os.path.exists(model_file + '.onnx') or os.path.getmtime(model_file + '.onnx') < os.path.getmtime(model_file):
            logging.info('Exporting model to onnx to run on using openvino on Intel cpu')
            torch.onnx.export(model.module, {'input': dummy_input}, model_file + '.onnx')

        if not os.path.exists(model_file + '.xml') or os.path.getmtime(model_file + '.xml') < os.path.getmtime(model_file):
            logging.info('Converting onnx model to openvino ...')
            mo_cmd = f'/groups/branson/home/kabram/.local/bin/mo --input_model {model_file}.onnx --compress_to_fp16 --output_dir {os.path.dirname(model_file)}'
            mo_result = os.system(mo_cmd)

        # ie = Core()
        # model_onnx = ie.read_model(model_file + '.onnx')
        # model_onnx = ie.compile_model(model_onnx, device_name='CPU')

        ie1 = Core()
        model_ir = ie1.read_model(model=model_file + '.xml')
        compiled_model_ir = ie1.compile_model(model_ir, device_name='CPU')

        return compiled_model_ir


    def restore(self, model_file,model, opt=None, sched=None):
        if model_file is None:
            with open(self.get_ckpt_file(),'r') as cf:
                prev_models = json.load(cf)
            model_file = prev_models[-1]
        logging.info('Loading model from {}'.format(model_file))
        if torch.cuda.is_available():
            ckpt = torch.load(model_file)
        else:
            ckpt = torch.load(model_file,map_location=torch.device('cpu'))
        if 'state_dict' in ckpt.keys():
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt['model_state_params'], strict=False)
        if self.conf.use_openvino:
            model = self.convert_openvino(model,model_file)
        if opt is not None:
            opt.load_state_dict(ckpt['optimizer_state_params'])
            for state in opt.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor):
                        state[k] = v.to(self.device)
        if sched is not None:
            sched.load_state_dict(ckpt['sched_state_params'])
        if 'meta' in ckpt:
            meta = ckpt['meta']
            if 'iter' in meta:
                start_at = meta['iter']
            else:
                raise RuntimeError('ckpt[\'meta\'] exists but ckpt[\'meta\'][\'iter\'] does not')
        elif 'step' in ckpt:                                           
            start_at = ckpt['step'] + 1
        else:
            raise RuntimeError('Cannot determine what iteration to start at from ckpt')        
        self.restore_td(start_at)
        return model, start_at


    def init_td(self):
        # initialize trianing info
        ex_td_fields = []
        for t_f in self.td_fields:
            ex_td_fields.append('val_' + t_f)
            ex_td_fields.append('train_' + t_f)
        ex_td_fields.extend(['step','l_rate'])
        train_info = OrderedDict()
        for t_f in ex_td_fields:
            train_info[t_f] = []
        self.train_info = train_info


    def restore_td(self, start_at=-1):
        # restore training info
        train_data_file = self.get_td_file().replace('\\', '/')
        with open(train_data_file, 'rb') as td_file:
            if sys.version_info.major == 3:
                in_data = pickle.load(td_file, encoding='latin1')
            else:
                in_data = pickle.load(td_file)

            if not isinstance(in_data, dict):
                train_info, load_conf = in_data
                logging.info('Parameters that do not match for {:s}:'.format(train_data_file))
                PoseTools.compare_conf(self.conf, load_conf)
            else:
                logging.warning("No config was stored for base. Not comparing conf")
                train_info = in_data
        if start_at > 0: # remove entries for step > start_at
            step = train_info['step'][:] # copy the list
            for k in train_info.keys():
                train_info[k] = [train_info[k][ix] for ix in range(len(step)) if step[ix]<= start_at]

        self.train_info = train_info


    def save_td(self):
        train_data_file = self.get_td_file()

        with open(train_data_file, 'wb') as td_file:
            pickle.dump([self.train_info, self.conf], td_file, protocol=2)
        json_data = {}
        for x in self.train_info.keys():
            json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
        with open(train_data_file+'.json','w') as json_file:
            json.dump(json_data, json_file)


    def update_td(self, cur_dict):
        # update training info
        if len(self.train_info) == 0:
            for k in cur_dict.keys():
                self.train_info[k] = []
        for k in cur_dict.keys():
            self.train_info[k].append(cur_dict[k])
        print_train_data(cur_dict)


    def compute_train_data(self,inputs,net,loss):
        labels = self.create_targets(inputs)
        with torch.no_grad():
            output = net(inputs)
        loss_val = loss(output,labels).sum().item()
        dist = self.compute_dist(output,labels)
        return {'cur_loss':loss_val, 'cur_dist':dist}


    def compute_dist(self,output,labels):
        return np.nan

    def create_data_gen(self,debug=False,pin_mem=False):
        if self.conf.db_format == 'tfrecord':
            return self.create_tf_data_gen(debug=debug)
        elif self.conf.db_format =='coco':
            return self.create_coco_data_gen(debug=debug,pin_mem=pin_mem)
        else:
            assert  False, 'Unknown data format type'


    def create_tf_data_gen(self, debug=False,**kwargs):
        assert False, 'TFRecordDataset unavailable'
        # conf = self.conf
        # train_tfn = lambda f: decode_augment(f,conf,True)
        # val_tfn = lambda f: decode_augment(f,conf,False)
        # trntfr = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        # valtfr = trntfr
        # xx = tf.python_io.tf_record_iterator(trntfr)
        # len_db = sum([1 for x in xx])
        # queue_sz = min(len_db,300)
        # # valtfr = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
        # if not os.path.exists(valtfr):
        #     logging.info('Validation data set doesnt exist. Using train data set for validation')
        #     valtfr = trntfr
        # train_dl_tf = TFRecordDataset(trntfr,None,None,transform=train_tfn,shuffle_queue_size=queue_sz)
        # val_dl_tf = TFRecordDataset(valtfr,None,None,transform=val_tfn)
        # self.train_loader_raw = train_dl_tf
        # self.val_loader_raw = val_dl_tf
        # num_workers = 0 if debug else 16
        # args = {
        #     'batch_size': self.conf.batch_size,
        #     'pin_memory': True,
        #     'drop_last': True,
        # }

        # if num_workers > 0:
        #     #try:
        #     #    self.train_dl = torch.utils.data.DataLoader(train_dl_tf, **args,num_workers=num_workers,worker_init_fn=lambda id: np.random.seed(id))
        #     #except:
        #     logging.warning(f'Could not create torch DataLoader with num_workers = {num_workers}, trying with 0')
        #     self.train_dl = torch.utils.data.DataLoader(train_dl_tf, **args, num_workers=0)
        # else:
        #     self.train_dl = torch.utils.data.DataLoader(train_dl_tf, **args, num_workers=0)

        # self.val_dl = torch.utils.data.DataLoader(val_dl_tf, **args)

        # self.train_iter = iter(self.train_dl)
        # self.val_iter = iter(self.val_dl)


    def create_coco_data_gen(self, debug=False, pin_mem=True, **kwargs):
        conf = self.conf
        trnjson = os.path.join(conf.cachedir, conf.trainfilename) + '.json'
        valjson = os.path.join(conf.cachedir, conf.valfilename) + '.json'
        train_dl_coco = coco_loader(conf,trnjson,True)
        if os.path.exists(valjson) and (len(PoseTools.json_load(valjson)['annotations'])>0):
            val_dl_coco = coco_loader(conf,valjson,False)
        else:
            logging.info('Val json file doesnt exist or is empty. Using training file for validation')
            val_dl_coco = coco_loader(conf,trnjson,False)
        self.train_loader_raw = train_dl_coco
        self.val_loader_raw = val_dl_coco

        num_workers = 0 if debug else 2
        logging.info('Number of DataLoader workers: %d', num_workers)

        self.train_dl = torch.utils.data.DataLoader(train_dl_coco, 
                                                    batch_size=self.conf.batch_size, 
                                                    pin_memory=pin_mem,
                                                    drop_last=True,
                                                    num_workers=num_workers,
                                                    shuffle=True,
                                                    worker_init_fn=dataloader_worker_init_fn)
        self.val_dl = torch.utils.data.DataLoader(val_dl_coco, batch_size=self.conf.batch_size, pin_memory=pin_mem, drop_last=True)
        self.train_iter = iter(self.train_dl)
        self.val_iter = iter(self.val_dl)

    def next_data(self, dtype):
        if dtype == 'train':
            it = self.train_iter
        else:
            it = self.val_iter

        try:
            ndata = next(it)
        except StopIteration:
            if dtype == 'train':
                self.train_epoch += 1
                if self.use_hard_mining and (self.step[0]/self.step[1]>0.001) and self.train_epoch>3:
                    wts = self.train_loader_raw.ex_wts
                    pcs = np.percentile(wts.numpy(),[5,95])
                    wts = torch.clamp(wts,pcs[0],pcs[1])
                    wt_range = 10.
                    wts = wts-wts.min()
                    wts = wts/wts.max()
                    wts = wts*(wt_range-1) + 1
                    train_sampler = torch.utils.data.WeightedRandomSampler(wts,self.train_loader_raw.len)
                    shuffle = False
                else:
                    train_sampler = None
                    shuffle = True if self.conf.db_format == 'coco' else False

                self.train_dl = torch.utils.data.DataLoader(
                    self.train_loader_raw, 
                    batch_size=self.conf.batch_size, 
                    pin_memory=True,
                    drop_last=True, 
                    num_workers=2,
                    sampler=train_sampler,
                    shuffle=shuffle,
                    worker_init_fn=partial(dataloader_worker_init_fn, epoch=self.train_epoch))
                self.train_iter = iter(self.train_dl)
                ndata = next(self.train_iter)
            else:
                self.val_dl = torch.utils.data.DataLoader(self.val_loader_raw, batch_size=self.conf.batch_size, pin_memory=True, drop_last=True)
                self.val_iter = iter(self.val_dl)
                ndata = next(self.val_iter)
        return ndata

    def create_targets(self, inputs):
        locs = inputs['locs']
        return PoseTools.create_label_images(locs,self.conf.imsz,1,self.conf.label_blur_rad)

    def create_optimizer(self, model, base_lr):
        return torch.optim.Adam(model.parameters(),lr=base_lr)

    def create_lr_sched(self,opt,training_iters,base_lr,step_lr,lr_drop_step_frac):
        if step_lr:
            def lambda_lr(x):
                if x < (1-lr_drop_step_frac)*training_iters:
                    return 1.
                elif x > 0.95*training_iters:
                    return 0.01
                else:
                    return 0.1

            # lambda_lr = lambda x: 0.1 if x > (1-lr_drop_step_frac)*training_iters else 1.
        else:
            lambda_lr = lambda x: self.conf.gamma ** (x/self.conf.decay_steps)

        return torch.optim.lr_scheduler.LambdaLR(opt,lambda_lr)

    def train(self, model, loss, opt, lr_sched, n_steps, start_at=0):

        save_start = time.time()
        clip_gradients = self.conf.get('clip_gradients', True)
        start = time.time()

        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        for step in range(start_at,n_steps):
            # gc.collect()
            self.step = [step,n_steps]
            a = time.time()
            inputs = self.next_data('train')
            l = time.time()
            labels = self.create_targets(inputs)
            t = time.time()
            opt.zero_grad()
            outputs = model(inputs)
            o = time.time()
            # valid = torch.any(torch.all(inputs['locs'] > -1000, dim=3), dim=2)
            # if not torch.all(torch.any(valid, dim=1)):
            #     print('Some inputs dont have any labels')
            #     continue

            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            loss_val = loss(outputs,labels)
            if self.use_hard_mining:
                n_ex = torch.any(inputs['locs'][:,:,:,0]>-10000,axis=-1).sum(axis=-1).cpu().numpy().copy()
                n_ex = np.maximum(n_ex,1)
                self.train_loader_raw.update_wts(inputs['item'].numpy(),loss_val.detach().cpu().numpy().copy()/n_ex)
            lo = time.time()
            # print(prof)
            loss_val.sum().backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            b = time.time()

            opt.step()
            lr_sched.step()
            op = time.time()
            # print('Timings Load:{:.2f}, target:{:.2f} fwd:{:.2f} loss:{:0.2f} bkwd:{:.2f} op:{:.2f}'.format(l-a,t-l,o-t,lo-t,b-lo,op-b))

            if self.conf.save_time is None:
                if (step % self.conf.save_step == 0) & (step>0):
                    self.save(step, model, opt, lr_sched)
            else:
                if ((time.time() - save_start) > self.conf.save_time*60) or step==0:
                    save_start = time.time()
                    self.save(step,model,opt,lr_sched)

            if (step+1) % self.conf.display_step == 0:
                en = time.time()
                logging.info('Time required to train:{}'.format(en-start))
                start = en
                train_in = self.next_data('train')
                train_dict = self.compute_train_data(train_in, model, loss)
                train_loss = train_dict['cur_loss']
                train_dist = train_dict['cur_dist']
                val_in = self.next_data('val')
                val_dict = self.compute_train_data(val_in, model, loss)
                val_loss = val_dict['cur_loss']
                val_dist = val_dict['cur_dist']
                cur_dict = OrderedDict()
                cur_dict['val_dist'] = val_dist
                cur_dict['train_dist'] = train_dist
                cur_dict['train_loss'] = train_loss
                cur_dict['val_loss'] = val_loss
                cur_dict['step'] = step + 1
                cur_dict['l_rate'] = lr_sched.get_last_lr()[0]
                self.update_td(cur_dict)
                self.save_td()

        logging.info("Optimization Finished!")
        self.save(n_steps, model, opt, lr_sched)

    def train_wrapper(self, restore=False, model_file=None,debug=False):
        model = self.create_model()
        training_iters = self.conf.dl_steps        
        learning_rate = self.conf.get('learning_rate_multiplier',1.)*self.conf.get('mdn_base_lr',0.0001)
        lr_drop_step_frac = self.conf.get('lr_drop_step', 0.15)
        step_lr = self.conf.get('step_lr', True)
        opt = self.create_optimizer(model,learning_rate)
        sched = self.create_lr_sched(opt,training_iters,learning_rate,step_lr,lr_drop_step_frac)

        logging.info('Using {} GPUS!'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

        if model_file is None:
            if restore:
                model_file = self.get_latest_model_file()
            if model_file is None:
                start_at = 0
                self.init_td()
            else:
                _, start_at = self.restore(model_file, model, opt, sched)
        else:
            try:
                if torch.cuda.device_count()==0:
                    ckpt =torch.load(model_file,map_location=torch.device('cpu'))
                else:
                    ckpt = torch.load(model_file)
                if 'state_dict' in ckpt.keys():
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt['model_state_params'])
                logging.info('Inititalizing model weights from {}'.format(model_file))
            except Exception as e:
                logging.info(f'Could not initialize model weights from {model_file}')
                logging.info(e)
            start_at = 0
            self.init_td()

        model.to(self.device)
        self.create_data_gen()

        if self.conf.get('use_dataset_stats_norm',False):
            all_ims = []
            for ndx in range(50):
                for skip in range(40):
                    cur_i, train_loader = self.next_data('train')
                all_ims.append(cur_i['images'].cpu().numpy())
            all_ims = np.concatenate(all_ims,0)
            im_mean = all_ims.mean((0,2,3),keepdims=True)
            im_std = all_ims.std((0,2,3),keepdims=True)
            model.module.im_mean = torch.tensor(im_mean[0,...].astype('single'))
            model.module.im_std = torch.tensor(im_std[0,...].astype('single'))

        self.train(model,self.loss,opt,sched,training_iters,start_at)

    def loss(self,output, labels):
        torch.nn.MSELoss(output-labels)

    def create_model(self):
        # Inherit this to create the model
        assert False, 'Inherit this function'

    def get_pred_fn(self, model_file):
        # Inherit this to create the model
        assert False, 'Inherit this function'

    def to_numpy(self, t):
        if type(t) is list or type(t) is tuple:
            return [self.to_numpy(tt) for tt in t]
        else:
            return t.detach().cpu().numpy()

    def diagnose(self, ims, out_file=None, **kwargs):
        model_file = kwargs.pop('model_file')
        pred_fn, close_fn, model_file = self.get_pred_fn(model_file,**kwargs)
        ret_dict = pred_fn(ims,retrawpred=True)
        conf = self.conf

        if out_file is None:
            out_file = os.path.join(conf.cachedir,'diagnose_' + PoseTools.get_datestr())

        with open(out_file,'wb') as f:
            pickle.dump({'ret_dict':ret_dict,'conf':conf,'ims':ims},f)

        close_fn()
        return ret_dict

