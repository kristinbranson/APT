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
import tensorflow as tf
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import argparse
import bsub
import copy
import subprocess
import nvidia_smi
import multiprocessing as mp
import time
import threading

# auto reload modules
# %load_ext autoreload
# %autoreload 2

assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
device_id = torch.cuda.current_device()
nvidia_smi.nvmlInit()

# %%
# parameters
cfglistfile = 'cfgs/cfgfilelist.csv'
stagenames = ['first','second']
locfilestr = 'loc.json'
base_cfg_dir = 'cfgs'
cachedir = base_cfg_dir
gpu_queue = 'gpu_a100'
ncores = 4
condaenv = 'APT'
NKPTS = 10
NITERS = 10
CRASHPROOF = True
NVIDIASMI = True

# %%
# read names of config files for all tracker types
cfgfiles = []
with open(cfglistfile,'r') as f:
    header = f.readline()
    # split by commas
    headers = header.strip().split(',')
    while True:
        line = f.readline()
        if not line:
            break
        parts = line.strip().split(',')
        if len(parts) == 0:
            continue
        assert len(parts) == len(headers), "Number of parts does not match number of headers"
        info = {headers[i]: parts[i] for i in range(len(parts))}
        cfgfiles.append(info)

# %%

def reset_memory_stats(device_id=0):
    """Reset GPU memory stats with proper error handling"""
    gc.collect()
    try:
        # First try to clear cache
        torch.cuda.empty_cache()
        
        # Then synchronize
        torch.cuda.synchronize()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error during reset: {e}")
            print("GPU may be in error state. Attempting recovery...")
            
            # Try to recover
            try:
                torch.cuda.empty_cache()
                gc.collect()
                
                # Try to reinitialize CUDA
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    torch.cuda.set_device(device)
                    torch.cuda.synchronize()
                    print("Recovery successful")
                    return True
            except:
                print("Recovery failed - GPU needs manual reset")
                return False
        raise

    tf.keras.backend.clear_session()  # Clear TF memory
    tf.config.experimental.reset_memory_stats(f'GPU:{device_id}')

def get_memory_usage(device_id=0):
    torch_peak_memory = torch.cuda.max_memory_allocated()
    torch_peak_reserved = torch.cuda.max_memory_reserved()
    tf_memory_info = tf.config.experimental.get_memory_info(f'GPU:{device_id}')
    return max(torch_peak_memory, tf_memory_info['peak']), torch_peak_reserved

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


def find_csv_files(conf, nettype_code, savedir=None, mintimestamp=None):
    if savedir is None:
        savedir = conf.cachedir
    csvfiles = [f for f in os.listdir(savedir) if f.startswith(f'network_memory_profile_{nettype_code}_') and f.endswith('.csv')]
    timestamps = [f[len(f'network_memory_profile_{nettype_code}_'):-len('.csv')] for f in csvfiles]
    csvfiles = [os.path.join(conf.cachedir,f) for f in csvfiles]
    if mintimestamp is not None:
        csvfiles = [csvfiles[i] for i in range(len(timestamps)) if timestamps[i] >= mintimestamp]
        timestamps = [timestamps[i] for i in range(len(timestamps)) if timestamps[i] >= mintimestamp]
    order = np.argsort(timestamps)
    csvfiles = [csvfiles[i] for i in order]
    timestamps = [timestamps[i] for i in order]
    return csvfiles, timestamps

def load_csv_file(csvfile):
    memory_allocated = {}
    memory_reserved = {}
    with open(csvfile,'r') as f:
        header = f.readline()
        flds = header.strip().split(',')
        try:
            bszi = flds.index('batch_size')
            imszi = flds.index('image_size')
            allocatedi = flds.index('peak_allocated_bytes')
            reservedi = flds.index('peak_reserved_bytes')
        except ValueError as e:
            print(f'Error parsing header in file {csvfile}: {e}')
            return memory_allocated, memory_reserved
        for line in f:
            parts = line.strip().split(',')
            assert len(parts) == len(flds), f"Error parsing line: {line} in file {csvfile}"
            bsz = int(parts[bszi])
            imsz = int(parts[imszi])
            memory_allocated[(bsz,imsz)] = float(parts[allocatedi])
            memory_reserved[(bsz,imsz)] = float(parts[reservedi])
    return memory_allocated, memory_reserved

def load_csv_files(csvfiles):
    memory_allocated = {}
    memory_reserved = {}
    allnan = True
    for csvfile in csvfiles:
        memory_allocated_curr, memory_reserved_curr = load_csv_file(csvfile)

        # copy to main dict
        nlines = 0
        ndatapoints = 0
        for (bsz,imsz) in memory_allocated_curr.keys():
            alloc = memory_allocated_curr[(bsz,imsz)]
            reserv = memory_reserved_curr[(bsz,imsz)]
            # add to dict if non-nan or nor already present
            if not np.isnan(alloc) or (bsz,imsz) not in memory_allocated:
                memory_allocated[(bsz,imsz)] = alloc
            if not np.isnan(reserv) or (bsz,imsz) not in memory_reserved:
                memory_reserved[(bsz,imsz)] = reserv 
            if not np.isnan(alloc):
                allnan = False
                ndatapoints += 1
            nlines += 1
        print(f'  Read {ndatapoints} non-nan results, {nlines} lines from file {csvfile}')
    
    return memory_allocated, memory_reserved, allnan

# %%

def profile_memory_usage_single(conf, args, net_type, batchsize_curr, imsize_curr):

    restore = None
    model_file = None    
    
    # copy conf
    conf = copy.deepcopy(conf)

    conf.imsz = (imsize_curr, imsize_curr)
    conf.batch_size = batchsize_curr
    conf.multi_frame_sz = (imsize_curr, imsize_curr)
    parentdir = os.path.dirname(conf.cachedir)
    conf.cachedir = os.path.join(parentdir, f'cache_bs{batchsize_curr}_im{imsize_curr}')
    os.makedirs(conf.cachedir, exist_ok=True)
    
    # Reset GPU memory stats and track high watermark
    reset_memory_stats()

    # train for a small number of iterations
    # Completely suppress all output
    create_dummy_training_data(conf, (imsize_curr,imsize_curr), batchsize_curr*NITERS)
    
    try:
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                apt.train_other_core(net_type, conf, args, restore, model_file)
    except torch.cuda.OutOfMemoryError:
        print(f"!!!OOM for batch size {batchsize_curr}, image size {imsize_curr}")
        peak_memory = np.nan
        peak_reserved = np.nan
    except Exception as e:
        print(f"!!!Failed for batch size {batchsize_curr}, image size {imsize_curr}: {e}")
        peak_memory = np.nan
        peak_reserved = np.nan
    else:
        peak_memory,peak_reserved = get_memory_usage()
    
    return peak_memory, peak_reserved

def run_subprocess(cmd, conn, timeout):
    """Run subprocess and send result through pipe"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        conn.send({
            'result': result,
            'success': True
        })
    except subprocess.TimeoutExpired as e:
        conn.send({
            'result': result,
            'error': 'timeout',
            'exception': str(e),
            'success': False
        })
    except Exception as e:
        conn.send({
            'result': result,
            'error': 'exception',
            'exception': str(e),
            'success': False
        })
    finally:
        conn.close()
        
def run_and_profile_memory(cmd,device=0,timeout=300):

    # Create pipe
    parent_conn, child_conn = mp.Pipe()
    
    # Start subprocess in separate process
    process = mp.Process(target=run_subprocess, args=(cmd, child_conn, timeout))
    process.start()
    
    max_mem = 0
    while process.is_alive():
        time.sleep(1)

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        mem = info.used
        max_mem = max(mem,max_mem)
        
    process.join()
    if parent_conn.poll():
        msg = parent_conn.recv()
        result = msg['result']
        if msg['success'] == False:
            print(f"Subprocess failed: error: {msg['error']}, exception: {msg['exception']}")
    else:
        result = None

    return max_mem,result

def run_in_subprocess(cmd,timeout=300):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)        
    return result

def process_single_output(result):
    # Parse result from stdout - should be "peak_memory,peak_reserved" on last line
    lines = result.stdout.strip().split('\n')
    assert lines and ',' in lines[-1], "Could not parse output"
    parts = lines[-1].split(',')
    assert len(parts) == 2, "Could not parse output"
    peak_memory = float(parts[0])
    peak_reserved = float(parts[1])
    return peak_memory, peak_reserved

def profile_memory_usage_matrix(cfgi, conf, args, net_type, batchsizes_try, imsizes_try, outfile, crashproof=CRASHPROOF, nvidiasmi=NVIDIASMI, restart=False, mintimestamp=None, recomputenans=False):

    nbatchsizes = len(batchsizes_try)
    nimsizes = len(imsizes_try)
    peak_memory = np.zeros((nbatchsizes,nimsizes))
    peak_reserved = np.zeros((nbatchsizes,nimsizes))
    peak_memory[:,:] = np.nan
    peak_reserved[:,:] = np.nan

    memory_allocated_pre = {}
    memory_reserved_pre = {}    
    if restart:
        # find existing csv files for this net type
        csvfiles, timestamps = find_csv_files(conf, net_type, mintimestamp=mintimestamp)
        if len(csvfiles) > 0:
            print(f"Found {len(csvfiles)} existing csv files for this net type, loading data")
            memory_allocated_pre, memory_reserved_pre, allnan = load_csv_files(csvfiles)                

    fid = open(outfile,'w')
    fid.write('batch_size,image_size,peak_allocated_bytes,peak_reserved_bytes\n')
    # close file
    fid.close()

    for batchsizei in range(len(batchsizes_try)):
        for imsizei in range(len(imsizes_try)):
            batchsize_curr = int(batchsizes_try[batchsizei])
            imsize_curr = int(imsizes_try[imsizei])
            
            if (batchsize_curr, imsize_curr) in memory_allocated_pre and (not recomputenans or not np.isnan(memory_allocated_pre[(batchsize_curr, imsize_curr)])):
                peak_memory[batchsizei,imsizei] = memory_allocated_pre[(batchsize_curr, imsize_curr)]
                peak_reserved[batchsizei,imsizei] = memory_reserved_pre[(batchsize_curr, imsize_curr)]
                print(f"Skipping batch size {batchsize_curr}, image size {imsize_curr} - already have data: Peak allocated: {peak_memory[batchsizei,imsizei] / 1024**3:.2f} GB, Peak reserved: {peak_reserved[batchsizei,imsizei] / 1024**3:.2f} GB")
            else:
                print(f"\nTrying batch size {batchsize_curr}, image size {imsize_curr}x{imsize_curr}")
                if crashproof:
                    # run in subprocess with timeout
                    cmd = [sys.executable, __file__, 'profilesingle', str(cfgi), str(imsize_curr), str(batchsize_curr)]
                    try:
                        if nvidiasmi:
                            maxmem,result = run_and_profile_memory(cmd,device=device_id,timeout=300)
                        else:
                            result = run_in_subprocess(cmd,timeout=300)
                            maxmem = np.nan
                        if result is None:
                            print(f"!!!No result returned for batch size {batchsize_curr}, image size {imsize_curr}")
                            peak_memory[batchsizei,imsizei] = maxmem
                            peak_reserved[batchsizei,imsizei] = np.nan
                            continue
                        if result.stderr:
                            print(result.stderr)
                        if result.stdout:
                            print(result.stdout)
                        if result.returncode != 0:
                            print(f"Subprocess failed with return code {result.returncode}: {result.stderr}")
                            peak_memory[batchsizei,imsizei] = maxmem
                            peak_reserved[batchsizei,imsizei] = np.nan
                            continue
                        # Parse result from stdout - should be "peak_memory,peak_reserved" on last line
                        peak_memory_curr, peak_reserved_curr = process_single_output(result)
                        if not np.isnan(maxmem):
                            peak_memory_curr = max(peak_memory_curr, maxmem)
                        peak_memory[batchsizei,imsizei] = peak_memory_curr
                        peak_reserved[batchsizei,imsizei] = peak_reserved_curr
                    except Exception as e:
                        print(f"!!!Failed for batch size {batchsize_curr}, image size {imsize_curr}: {e}")
                        peak_memory[batchsizei,imsizei] = np.nan
                        peak_reserved[batchsizei,imsizei] = np.nan
                else:
                    peak_memory[batchsizei,imsizei], peak_reserved[batchsizei,imsizei] = profile_memory_usage_single(conf, args, net_type, batchsize_curr, imsize_curr)

            print(f"  Peak allocated: {peak_memory[batchsizei,imsizei] / 1024**3:.2f} GB, Peak reserved: {peak_reserved[batchsizei,imsizei] / 1024**3:.2f} GB")
            # add this line to the output csv file
            fid = open(outfile,'a')
            fid.write(f"{batchsize_curr},{imsize_curr},{peak_memory[batchsizei,imsizei]},{peak_reserved[batchsizei,imsizei]}\n")
            fid.close()

    return peak_memory, peak_reserved 

# %%

def load_config(cfginfo):

    print(cfginfo)
    json_trn_file = os.path.join(base_cfg_dir, locfilestr)
    base_cfg_file = cfginfo['cfgfile']
    # remove deepnet from base_cfg_file if present
    if base_cfg_file.startswith('deepnet/'):
        base_cfg_file = base_cfg_file[len('deepnet/'):]
    stageidx = int(cfginfo['stageidx'])
    print(f'Using base config file: {base_cfg_file}, stage index: {stageidx}')
    cur_view = 0
    nettype_code = cfginfo['nettype']

    cfgdict = apt.load_config_file(base_cfg_file)
    trackerdata = cfgdict['TrackerData']
    if isinstance(trackerdata, list):
        trackerdata = trackerdata[stageidx]
    net_type = trackerdata['trnNetTypeString']
    stage = stagenames[stageidx]
    argv = [base_cfg_file, '-name', 'base_profile', '-json_trn_file', json_trn_file, '-stage', stage, '-conf_params', '-type', net_type, '-ignore_local', '1', '-cache', cachedir, 'train', '-use_cache', '-skip_db']
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
    
    if NKPTS is not None:
        conf.n_classes = NKPTS

    # set number of iterations to something small
    conf.dl_steps = NITERS
    conf.display_step = NITERS
    conf.save_step = NITERS
    conf.view = cur_view
    
    return conf, args, net_type, nettype_code

# %%

# debug what happens if images size doesn't match conf.imsz
if False:

    # train for a small number of iterations
    imsize_curr = 256
    batchsize_curr = 2
    create_dummy_training_data(conf, (imsize_curr,imsize_curr), batchsize_curr)
    conf.imsz = (imsize_curr*2, imsize_curr*2)
    conf.batch_size = batchsize_curr
    conf.multi_frame_sz = (imsize_curr*2, imsize_curr*2)
    apt.train_other_core(net_type, conf, args, restore, model_file)

# %%

def profile(cfgi0=0, cfgi1=None, batchsizes_try=None, imsizes_try=None, nvidiasmi=False, restart=False, mintimestamp=None, recomputenans=False):
    
    # use datetime to get current time
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')

    if cfgi1 is None:
        cfgi1 = len(cfgfiles)

    if batchsizes_try is None:
        batchsizes_try = 2**np.arange(6,dtype=int)
    else:
        batchsizes_try = np.atleast_1d(batchsizes_try).astype(int)

    if imsizes_try is None:
        imsizes_try = 2**np.arange(6,14,dtype=int)
    else:
        imsizes_try = np.atleast_1d(imsizes_try).astype(int)

    print('batch sizes to try: ', batchsizes_try)
    print('image sizes to try: ', imsizes_try)
        
    for cfgi in range(cfgi0,cfgi1):
        print(f'\nProfiling config file {cfgi+1} of {len(cfgfiles)}')    
        conf, args, net_type, nettype_code = load_config(cfgfiles[cfgi])
        outfile = os.path.join(conf.cachedir,f'network_memory_profile_{nettype_code}_{timestamp}.csv')
        print('Writing results to ', outfile)
        res = profile_memory_usage_matrix(cfgi, conf, args, net_type, batchsizes_try, imsizes_try, outfile, nvidiasmi=nvidiasmi, restart=restart, mintimestamp=mintimestamp, recomputenans=recomputenans)

        # save results to npy file
        peak_memory, peak_reserved = res
        npyfile = os.path.join(conf.cachedir,f'network_memory_profile_{nettype_code}_{timestamp}.npz')
        print('Saving results to ', npyfile)
        np.savez(npyfile, peak_memory=peak_memory, peak_reserved=peak_reserved, batchsizes_try=batchsizes_try, imsizes_try=imsizes_try)
        
def collect(savedir='network_memory_profile',mintimestamp=None):
    """
    collect()
    Collect results from all config files and save to single csv file per net type
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for cfgi in range(len(cfgfiles)):
        try:
            conf, args, net_type, nettype_code = load_config(cfgfiles[cfgi])
        except Exception as e:
            print(f"!!!Failed to load config for cfgi={cfgi}: {e}")
            continue
        
        print(f'\n\nCollecting results for net type {nettype_code}')
        # find all files in conf.cachedir that match network_memory_profile_{nettype_code}_{timestamp}.csv
        csvfiles_old, timestamps_old = find_csv_files(conf, nettype_code, savedir=savedir, mintimestamp=None)
        csvfiles_new,timestamps_new = find_csv_files(conf, nettype_code, mintimestamp=mintimestamp)
        csvfiles = csvfiles_old + csvfiles_new
        timestamps = timestamps_old + timestamps_new
        if len(csvfiles) == 0:
            print(f'!!!No csv files found for net type {nettype_code}, skipping')
            continue
        print(f'Found {len(csvfiles)} csv files for net type {nettype_code}')
        order = np.argsort(timestamps)
        csvfiles = [csvfiles[i] for i in order]
        timestamps = [timestamps[i] for i in order]
        memory_allocated, memory_reserved, allnan = load_csv_files(csvfiles)
        if allnan:
            print(f'!!!All entries are NaN for net type {nettype_code}, skipping')
            continue
        outcsvfile = os.path.join(savedir,f'network_memory_profile_{nettype_code}.csv')
        # sort keys
        keys = sorted(memory_allocated.keys())
        with open(outcsvfile,'w') as f:
            f.write('batch_size,image_size,peak_allocated_bytes,peak_reserved_bytes\n')
            for k in keys:
                bsz, imsz = k
                alloc = memory_allocated[k]
                reserv = memory_reserved[k] if k in memory_reserved else np.nan
                f.write(f'{bsz},{imsz},{alloc},{reserv}\n')
                
if __name__ == '__main__':
    
    # usage: python profile_network_size_kb.py
    # runs main function
    # python profile_network_size_kb.py profile cfgi imsize batchsize
    # just runs single profile_memory_usage_single for given cfgi, imsize, batchsize
    # python profile_network_size_kb.py profile --cfgi0 cfgi0 --cfgi1 cfgi1 --batchsizes 2,4,8 --imsizes 64,128,256
    # runs main function for given range of config files and batch sizes and image sizes
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, nargs='?', default='profile', help='Action to perform: profile or collect')
    parser.add_argument('cfgi', type=int, nargs='?', default=None, help='Index of config file to use (0-based)')
    parser.add_argument('imsize', type=int, nargs='?', default=None, help='Image size to use (assumes square images)')
    parser.add_argument('batchsize', type=int, nargs='?', default=None, help='Batch size to use')
    parser.add_argument('--cfgi0', type=int, default=0, help='Start index of config files to use (0-based)')
    parser.add_argument('--cfgi1', type=int, default=None, help='End index of config files to use (0-based, exclusive)')
    parser.add_argument('--batchsizes', type=str, default=None, help='Comma-separated list of batch sizes to try')
    parser.add_argument('--imsizes', type=str, default=None, help='Comma-separated list of image sizes to try')
    # add arguments for restarting
    parser.add_argument('--restart', action='store_true', help='Restart from existing csv files if present')
    parser.add_argument('--mintimestamp', type=str, default=None, help='Minimum timestamp (YYYYMMDDTHHMMSS) of csv files to use when restarting')
    parser.add_argument('--recomputenans', action='store_true', help='Recompute entries that are NaN when restarting')
    
    args = parser.parse_args()
    
    if args.action == 'collect':
        collect(mintimestamp=args.mintimestamp)
    elif args.action == 'profilesingle':
        assert args.cfgi is not None and args.imsize is not None and args.batchsize is not None, "cfgi, imsize, and batchsize must be provided for profilesingle action"
        print(f'Running single profile for cfgi={args.cfgi}, imsize={args.imsize}, batchsize={args.batchsize}')
        conf, args2, net_type, nettype_code = load_config(cfgfiles[args.cfgi])
        peak_memory, peak_reserved = profile_memory_usage_single(conf, args2, net_type, args.batchsize, args.imsize)
        print(f"{peak_memory},{peak_reserved}")
    elif args.action == 'profile':
        cfgi0 = args.cfgi0
        cfgi1 = args.cfgi1
        batchsizes_try = [int(x) for x in args.batchsizes.split(',')] if args.batchsizes else None
        imsizes_try = [int(x) for x in args.imsizes.split(',')] if args.imsizes else None
        profile(cfgi0=cfgi0, cfgi1=cfgi1, batchsizes_try=batchsizes_try, imsizes_try=imsizes_try, restart=args.restart, mintimestamp=args.mintimestamp, recomputenans=args.recomputenans)
    else:
        raise ValueError(f"Unknown action: {args.action}")