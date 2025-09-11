# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: APT
#     language: python
#     name: python3
# ---

# %%
import APT_interface as apt
import os
import numpy as np
import torch
import json
import sys
import gc
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# auto reload modules
# %load_ext autoreload
# %autoreload 2

assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."



# %%
# use datetime to get current time
timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')

# %%
import logging
logging.getLogger().setLevel(logging.INFO)

# %%
cfgfilestr = 'cfg.json'
locfilestr = 'loc.json'
cachedirstr = 'cache'
stagenames = ['first','second']

batchsizes_try = 2**np.arange(7,dtype=int)
imsizes_try = 2**np.arange(5,15,dtype=int)
print('batch sizes to try: ', batchsizes_try)
print('image sizes to try: ', imsizes_try)

# %%
base_cfg_dir = 'cfgs/ma_multifly_topdown_20260908'
cachedir = os.path.join(base_cfg_dir, cachedirstr)
json_trn_file = os.path.join(base_cfg_dir, locfilestr)
base_cfg_file = os.path.join(base_cfg_dir,cfgfilestr)
stageidx = 0
stage = stagenames[stageidx]
cur_view = 0


# %%
cfgdict = apt.load_config_file(base_cfg_file)
stage = 'first'

net_type = cfgdict['TrackerData'][stageidx]['trnNetTypeString']
argv = [base_cfg_file, '-name', 'base_profile', '-json_trn_file', json_trn_file, '-stage', stage, '-conf_params', '-type', net_type, '-ignore_local', '1', '-cache', 'cfgs/ma_multifly_topdown_20260908/cache', 'train', '-use_cache', '-skip_db']
args = apt.parse_args(argv)

conf_params = []
# create poseConfig
conf = apt.create_conf(cfgdict, 
                        cur_view, 
                        args.name, 
                        net_type=net_type, 
                        cache_dir=cachedir,
                        conf_params=conf_params,
                        json_trn_file=json_trn_file,
                        first_stage=stage=='first',
                        second_stage=stage=='second')

# set number of iterations to something small
conf.dl_steps = 10
conf.view = cur_view

model_file = None
restore = None


# %%
def create_dummy_training_data(conf,imsz,n,ncolors=3):
    
    train_filename = os.path.join(conf.cachedir, conf.trainfilename)
    os.makedirs(os.path.join(conf.cachedir, 'train'), exist_ok=True)
    
    nkpts = conf.n_classes
    
    skeleton = [[i, i + 1] for i in range(conf.n_classes - 1)]
    names = ['pt_{}'.format(i) for i in range(conf.n_classes)]
    categories = [{'id': 1, 'skeleton': skeleton, 'keypoints': names, 'super_category': 'fly', 'name': 'fly'}, {'id': 2, 'super_category': 'neg_box', 'name': 'neg_box'}]
    train_ann = {'images': [], 'info': [], 'annotations': [], 'categories': categories}
    
    train_ann = {'images': [], 'info': [], 'annotations': [], 'categories': categories}
    train_info = {'ndx': 0, 'ann_ndx': 0, 'imdir': os.path.join(conf.cachedir, 'train')}

    outfn = lambda data: apt.convert_to_coco(train_info, train_ann, data, conf)
    
    for i in range(n):
        cur_frame = np.random.randint(0,256,imsz+(ncolors,),dtype=np.uint8)
        cur_locs = np.zeros((1,nkpts,2))
        cur_occ = np.zeros((1,nkpts))
        for j in range(2):
            cur_locs[...,j] = np.random.uniform(0,imsz[j]-1,(nkpts,))
        info = [0,i,0] # this last 0 is weird
        minx = np.min(cur_locs[0,:,0])
        miny = np.min(cur_locs[0,:,1])
        maxx = np.max(cur_locs[0,:,0])
        maxy = np.max(cur_locs[0,:,1])
        cur_roi = np.array([[[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]])
        extra_roi = None
        data_out = [{'im': cur_frame, 'locs': cur_locs, 'info': info, 'occ': cur_occ, 'roi': cur_roi, 'extra_roi': extra_roi, 'max_n': conf.max_n_animals}]
        for curd in data_out:
            outfn(curd)
        
    with open(train_filename + '.json', 'w') as f:
        json.dump(train_ann, f)


# %%
def profile_memory_usage(conf, args, net_type, batchsizes_try, imsizes_try, outfile):

    fid = open(outfile,'w')
    fid.write('batch_size,image_size,peak_allocated_bytes,peak_reserved_bytes\n')
    # close file
    fid.close()

    restore = None
    model_file = None

    nbatchsizes = len(batchsizes_try)
    nimsizes = len(imsizes_try)
    peak_memory = np.zeros((nbatchsizes,nimsizes))
    peak_reserved = np.zeros((nbatchsizes,nimsizes))

    for batchsizei in range(len(batchsizes_try)):
        for imsizei in range(len(imsizes_try)):
            batchsize_curr = int(batchsizes_try[batchsizei])
            imsize_curr = int(imsizes_try[imsizei])
            print(f"\nTrying batch size {batchsize_curr}, image size {imsize_curr}x{imsize_curr}")
            idx = (batchsizei, imsizei)

            conf.imsz = (imsize_curr, imsize_curr)
            conf.batch_size = batchsize_curr
            conf.multi_frame_sz = (imsize_curr, imsize_curr)
            
            # Reset GPU memory stats and track high watermark
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete
            gc.collect()

            torch.cuda.reset_peak_memory_stats()
            
            # train for a small number of iterations
            # Completely suppress all output
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    create_dummy_training_data(conf, (imsize_curr,imsize_curr), batchsize_curr)
                    try:
                        apt.train_other_core(net_type, conf, args, restore, model_file)
                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM for batch size {batchsize_curr}, image size {imsize_curr}")
                        torch.cuda.empty_cache()  # Clean up after OOM
                        peak_memory[batchsizei,imsizei] = np.nan
                        peak_reserved[batchsizei,imsizei] = np.nan
                    except Exception as e:
                        print(f"  Failed for batch size {batchsize_curr}, image size {imsize_curr}: {e}")
                        peak_memory[batchsizei,imsizei] = np.nan
                        peak_reserved[batchsizei,imsizei] = np.nan
                    else:
                        peak_memory[batchsizei,imsizei] = torch.cuda.max_memory_allocated()
                        peak_reserved[batchsizei,imsizei] = torch.cuda.max_memory_reserved()
            
            print(f"  Peak allocated: {peak_memory[batchsizei,imsizei] / 1024**3:.2f} GB, Peak reserved: {peak_reserved[batchsizei,imsizei] / 1024**3:.2f} GB")
            # add this line to the output csv file
            fid = open(outfile,'a')
            fid.write(f"{batchsize_curr},{imsize_curr},{peak_memory[batchsizei,imsizei]},{peak_reserved[batchsizei,imsizei]}\n")
            fid.close()
    return peak_memory, peak_reserved 
        


# %%
torch.cuda.empty_cache()
torch.cuda.synchronize()  # Ensure all operations complete
gc.collect()

torch.cuda.reset_peak_memory_stats()

# train for a small number of iterations
imsize_curr = 256
batchsize_curr = 2
create_dummy_training_data(conf, (imsize_curr,imsize_curr), batchsize_curr)


conf.imsz = (imsize_curr*2, imsize_curr*2)
conf.batch_size = batchsize_curr
conf.multi_frame_sz = (imsize_curr*2, imsize_curr*2)


apt.train_other_core(net_type, conf, args, restore, model_file)


# %%
# outfile = f'network_memory_profile_{net_type}_{timestamp}.csv'
# add conf.mmdetect_net if net_type == 'detect_mmdetect'
# print('Writing results to ', outfile)
# peak_memory,peak_reserved = profile_memory_usage(conf, args, net_type, batchsizes_try, imsizes_try, outfile)
# # save results to npy file
# npyfile = f'network_memory_profile_{net_type}_{timestamp}.npz'
# print('Saving results to ', npyfile)
# np.savez(npyfile, peak_memory=peak_memory, peak_reserved=peak_reserved, batchsizes_try=batchsizes_try, imsizes_try=imsizes_try)
