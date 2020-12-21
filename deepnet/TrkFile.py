import numpy as np
import os
import pickle
import hdf5storage
import logging
import datetime
import h5py


def datetime2matlabdn(dt=datetime.datetime.now()):
  mdn=dt+datetime.timedelta(days=366)
  frac_seconds=(dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds/(24.0*60.0*60.0)
  frac_microseconds=dt.microsecond/(24.0*60.0*60.0*1000000.0)
  return mdn.toordinal()+frac_seconds+frac_microseconds

def convert(in_data,to_python):
  if type(in_data) in [list,tuple]:
    out_data=[]
    for i in in_data:
      out_data.append(convert(i,to_python))
  elif type(in_data) is dict:
    out_data={}
    for i in in_data.keys():
      out_data[i]=convert(in_data[i],to_python)
  elif in_data is None:
    out_data=None
  else:
    offset=-1 if to_python else 1
    out_data=in_data+offset
  return out_data

def to_mat(in_data):
  return convert(in_data,to_python=False)

def savemat_with_catch_and_pickle(filename,out_dict):
  try:
    # logging.info('Saving to mat file %s using hdf5storage.savemat'%filename)
    # sio.savemat(filename, out_dict, appendmat=False)
    hdf5storage.savemat(filename,out_dict,appendmat=False,truncate_existing=True)
  except Exception as e:
    logging.info('Exception caught saving mat-file {}: {}'.format(filename,e))
    logging.info('Pickling to {}...'.format(filename))
    with open(filename,'wb') as fh:
      pickle.dump(out_dict,fh)

def convert_to_mat_trk(in_pred,conf,start,end,trx_ids,has_trx_file=None):
  ''' Converts predictions to compatible trk format'''
  pred_locs=in_pred.copy()
  pred_locs=pred_locs[:,trx_ids,...]
  pred_locs=pred_locs[:(end-start),...]
  if pred_locs.ndim==4:
    pred_locs=pred_locs.transpose([2,3,0,1])
  else:
    pred_locs=pred_locs.transpose([2,0,1])
  if has_trx_file is None:
    has_trx_file=conf.has_trx_file
  if not has_trx_file:
    pred_locs=pred_locs[...,0]
  return pred_locs

def write_trk(out_file,pred_locs_in,extra_dict,start,end,trx_ids,conf,info,mov_file,has_trx_file=None):
  '''
  pred_locs is the predicted locations of size
  n_frames in the movie x n_Trx x n_body_parts x 2
  n_done is the number of frames that have been tracked.
  everything should be 0-indexed
  '''
  pred_locs=convert_to_mat_trk(pred_locs_in,conf,start,end,trx_ids,has_trx_file)
  pred_locs=to_mat(pred_locs)
  
  tgt=to_mat(np.array(trx_ids))  # target animals that have been tracked.
  # For projects without trx file this is always 1.
  ts_shape=pred_locs.shape[0:1]+pred_locs.shape[2:]
  ts=np.ones(ts_shape)*datetime2matlabdn()  # time stamp
  tag=np.zeros(ts.shape).astype('bool')  # tag which is always false for now.
  tracked_shape=pred_locs.shape[2]
  tracked=np.zeros([1,
                    tracked_shape])  # which of the predlocs have been tracked. Mostly to help APT know how much tracking has been done.
  tracked[0,:]=to_mat(np.arange(start,end))
  
  out_dict={'pTrk': pred_locs,
            'pTrkTS': ts,
            'expname': mov_file,
            'pTrkiTgt': tgt,
            'pTrkTag': tag,
            'pTrkFrm': tracked,
            'trkInfo': info}
  for k in extra_dict.keys():
    tmp=convert_to_mat_trk(extra_dict[k],conf,start,end,trx_ids,has_trx_file)
    if k.startswith('locs_'):
      tmp=to_mat(tmp)
    out_dict['pTrk'+k]=tmp
  
  # output to a temporary file and then rename to real file name.
  # this is because existence of trk file is a flag that tracking is done for
  # other processes, and writing may still be in progress when file discovered.
  out_file_tmp=out_file+'.tmp'
  savemat_with_catch_and_pickle(out_file_tmp,out_dict)
  if os.path.exists(out_file_tmp):
    os.rename(out_file_tmp,out_file)
  else:
    logging.exception("Did not successfully write output to %s"%out_file_tmp)