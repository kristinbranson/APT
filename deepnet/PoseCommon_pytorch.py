import torch
import os
import numpy as np
import json
import pickle
import sys
from collections import OrderedDict
import logging
import PoseTools
import tfdatagen
import time
import tensorflow.compat.v1 as tf
from tfrecord.torch.dataset import TFRecordDataset
import errno
import re
import gc
from torch import autograd
import time
autograd.set_detect_anomaly(True)

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
        ims = ims * features['mask']

    if 'max_n' in features.keys():
        n_max = features['max_n'][0]
    else:
        n_max = None

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
        locs = features['locs'].reshape([1,n_max,n_pts,2])

    if 'trx_ndx' not in features.keys():
        features['trx_ndx'] = np.array([0])


    features['info'] = np.array([features['expndx'][0],features['ts'][0],features['trx_ndx'][0]])


    ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.rescale)

    # convert CHW format
    ims = np.transpose(ims[0,...]/255.,[2,0,1])

    features['images'] = ims
    features['locs'] = locs[0,...]

    return features


def next_data(loader, dataset):
    try:
        ndata = next(loader)
    except StopIteration:
        loader = iter(dataset)
        ndata = next(loader)
    return ndata, loader

class PoseCommon_pytorch(object):

    def __init__(self,conf,name='deepnet',is_multi=False):
        self.conf = conf
        self.name = name
        self.is_multi = is_multi
        self.prev_models = []
        self.td_fields = ['dist','loss']
        conf.is_multi = is_multi

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print('CUDA Device not available. Using CPU!')

    def get_ckpt_file(self):
        return os.path.join(self.conf.cachedir,self.name + '_ckpt')

    def get_latest_model_file(self):
        ckpt_file = self.get_ckpt_file()
        if not os.path.exists(ckpt_file):
            model_file = None
        else:
            with open(ckpt_file,'r') as f:
                mf = json.load(f)
            if len(mf) > 1:
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
            if os.path.exists(self.prev_models[0]):
                os.remove(self.prev_models[0])
            _ = self.prev_models.pop(0)

        with open(self.get_ckpt_file(),'w') as cf:
            json.dump(self.prev_models,cf)

    def restore(self, model_file,model, opt=None, sched=None):
        if model_file is None:
            with open(self.get_ckpt_file(),'r') as cf:
                prev_models = json.load(cf)
            model_file = prev_models[-1]
        logging.info('Loading model from {}'.format(model_file))
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['model_state_params'])
        if opt is not None:
            opt.load_state_dict(ckpt['optimizer_state_params'])
            for state in opt.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor):
                        state[k] = v.to(self.device)
        if sched is not None:
            sched.load_state_dict(ckpt['sched_state_params'])
        start_at = ckpt['step'] + 1
        self.restore_td(start_at)
        return start_at


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
        output = net(inputs)
        loss_val = loss(output,labels).detach().item()
        dist = self.compute_dist(output,labels)
        return {'cur_loss':loss_val, 'cur_dist':dist}


    def compute_dist(self,output,labels):
        return np.nan


    def create_data_gen(self, **kwargs):
        conf = self.conf
        train_tfn = lambda f: decode_augment(f,conf,True)
        val_tfn = lambda f: decode_augment(f,conf,False)
        trntfr = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
        valtfr = os.path.join(conf.cachedir, conf.valfilename) + '.tfrecords'
        if not os.path.exists(valtfr):
            logging.info('Validation data set doesnt exist. Using train data set for validation')
            valtfr = trntfr
        train_dl_tf = TFRecordDataset(trntfr,None,None,transform=train_tfn,shuffle_queue_size=300)
        val_dl_tf = TFRecordDataset(valtfr,None,None,transform=val_tfn)
        train_dl = torch.utils.data.DataLoader(train_dl_tf, batch_size=self.conf.batch_size,pin_memory=True,drop_last=True,num_workers=16)
        val_dl = torch.utils.data.DataLoader(val_dl_tf, batch_size=self.conf.batch_size,pin_memory=True,drop_last=True)
        return [train_dl, val_dl]


    def create_targets(self, inputs):
        locs = inputs['locs']
        return PoseTools.create_label_images(locs,self.conf.imsz,1,self.conf.label_blur_rad)


    def create_optimizer(self, model, base_lr):
        return torch.optim.Adam(model.parameters(),lr=base_lr)

    def create_lr_sched(self,opt,training_iters,base_lr,step_lr,lr_drop_step_frac):
        if step_lr:
            lambda_lr = lambda x: 0.1 if x > (1-lr_drop_step_frac)*training_iters else 1.
        else:
            lambda_lr = lambda x: self.conf.gamma ** (x/self.conf.decay_steps)

        return torch.optim.lr_scheduler.LambdaLR(opt,lambda_lr)


    def train(self, data_loaders, model, loss, opt, lr_sched, n_steps, start_at=0):
        train_datagen = data_loaders[0]
        if len(data_loaders) > 1:
            val_datagen = data_loaders[1]
        else:
            val_datagen = data_loaders[0]

        train_loader = iter(train_datagen)
        val_loader = iter(val_datagen)
        save_start = time.time()
        clip_gradients = self.conf.get('clip_gradients', True)
        start = time.time()
        for step in range(start_at,n_steps):
            # gc.collect()
            a = time.time()
            inputs, train_loader = next_data(train_loader,train_datagen)
            l = time.time()
            opt.zero_grad()
            outputs = model(inputs)
            o = time.time()
            labels = self.create_targets(inputs)
            valid = torch.any(torch.all(labels > -1000, dim=3), dim=2)
            if not torch.all(torch.any(valid, dim=1)):
                print('Some inputs dont have any labels')
                continue

            t = time.time()
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            loss_val = loss(outputs,labels)
            lo = time.time()
            # print(prof)
            loss_val.backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            b = time.time()

            opt.step()
            lr_sched.step()
            op = time.time()
            # print('Timings Load:{:.2f}, target:{:.2f} fwd:{:.2f} loss:{:0.2f} bkwd:{:.2f} op:{:.2f}'.format(l-a,o-l,t-o,lo-t,b-lo,op-b))

            if self.conf.save_time is None:
                if step % self.conf.save_step == 0:
                    self.save(step, model, opt, lr_sched)
            else:
                if ((time.time() - save_start) > self.conf.save_time*60) or step==0:
                    save_start = time.time()
                    self.save(step,model,opt,lr_sched)

            if step % self.conf.display_step == 0:
                en = time.time()
                logging.info('Time required to train:{}'.format(en-start))
                start = en
                train_in, train_loader = next_data(train_loader, train_datagen)
                train_dict = self.compute_train_data(train_in, model, loss)
                train_loss = train_dict['cur_loss']
                train_dist = train_dict['cur_dist']
                val_in, val_loader = next_data(val_loader, val_datagen)
                val_dict = self.compute_train_data(val_in, model, loss)
                val_loss = val_dict['cur_loss']
                val_dist = val_dict['cur_dist']
                cur_dict = OrderedDict()
                cur_dict['val_dist'] = val_dist
                cur_dict['train_dist'] = train_dist
                cur_dict['train_loss'] = train_loss
                cur_dict['val_loss'] = val_loss
                cur_dict['step'] = step
                cur_dict['l_rate'] = lr_sched.get_last_lr()[0]
                self.update_td(cur_dict)
                self.save_td()

        logging.info("Optimization Finished!")
        self.save(n_steps, model, opt, lr_sched)

    def train_wrapper(self, restore=False):
        model = self.create_model()
        training_iters = self.conf.dl_steps
        learning_rate = self.conf.get('learning_rate_multiplier',1.)*self.conf.get('mdn_base_lr',0.0001)
        lr_drop_step_frac = self.conf.get('lr_drop_step', 0.15)
        step_lr = self.conf.get('step_lr', True)
        opt = self.create_optimizer(model,learning_rate)
        sched = self.create_lr_sched(opt,training_iters,learning_rate,step_lr,lr_drop_step_frac)

        logging.info('Using {} GPUS!'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

        if restore:
            model_file = self.get_latest_model_file()
        else:
            model_file = None
        if model_file is None:
            start_at = 0
            self.init_td()
        else:
            start_at = self.restore(model_file, model, opt, sched)

        model.to(self.device)
        data_loaders = self.create_data_gen()
        self.train(data_loaders,model,self.loss,opt,sched,training_iters,start_at)

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