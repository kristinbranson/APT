import numpy as np
import numpy.random as random
import scipy.optimize as opt
import TrkFile
import APT_interface as apt
import logging
import os
import scipy
import pickle

# for now I'm just using loadmat and savemat here
# when/if the format of trk files changes, then this will need to get fancier

from tqdm import tqdm
import torch
from torchvision import models
from torch import optim
import torch.nn.functional as F
import PoseTools
import movies
import tempfile
import copy
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

# for debugging
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import time

def match_frame(pcurr, pnext, idscurr, params, lastid=np.nan, maxcost=None,force_match=False):
  """
  match_frame(pcurr,pnext,idscurr,params,lastid=np.nan)
  Uses the Hungarian algorithm to match targets tracked in the current
  frame with targets detected in the next frame. The cost of
  assigning target i to detection j is the L1 error between the
  2*nlandmarks dimensional vectors normalized by the number of landmarks.
  The cost of a trajectory birth or death is params['maxcost']/2. Thus,
  it is preferable to kill one trajectory and create another if
  the matching error is > params['maxcost']
  Inputs:
  d x nlandmarks x ncurr positions of landmarks of nnext animals
  detected in the next frame
  idscurr: ncurr array, integer ids of the animals tracked in the
  current frame
  params: dictionary of parameters.
  lastid: (optional) scalar, last id used in tracking so far, if there are
  trajectory births, they will start with id lastid+1
  Outputs:
  idsnext: nnext array, integer ids assigned to animals in next frame
  Parameters:
  params['maxcost']: The cost of a trajectory birth or death is
  params['maxcost']/2. Thus, it is preferable to kill one trajectory
  and create another if the matching error is > params['maxcost'].
  params['verbose']: Whether to print out information
  """
  
  # pcurr: d x nlandmarks x ncurr
  # pnext: d x nlandmarks x nnext
  
  # check sizes
  nlandmarks = pcurr.shape[0]
  d = pcurr.shape[1]
  ncurr = pcurr.shape[2]
  nnext = pnext.shape[2]
  assert pnext.shape[0] == nlandmarks, \
    'N landmarks do not match, curr = %d, next = %d' % (nlandmarks, pnext.shape[0])
  assert pnext.shape[1] == d, \
    'Dimensions do not match, curr = %d, next = %d' % (d, pnext.shape[1])
  if maxcost is None:
    maxcost = params['maxcost']
  
  # construct the cost matrix
  # C[i,j] is the cost of matching curr[i] and next[j]
  C = np.zeros((ncurr+nnext, ncurr+nnext))
  C[:] = maxcost / 2.
  C[ncurr:, nnext:] = 0
  pcurr = np.reshape(pcurr, (d * nlandmarks, ncurr, 1))
  pnext = np.reshape(pnext, (d * nlandmarks, 1, nnext))
  C1 = np.sum(np.abs(pcurr-pnext), axis=0)/nlandmarks
  C[:ncurr, :nnext] = np.reshape(C1, (ncurr, nnext))

  if not force_match:
    # Don't do the ratio to second lowest match if force_match is on. This is used when estimating the maxcost parameter.

    # If a current detection has 2 matches then break the tracklet
    for x1 in range(ncurr):
      if np.all(np.isnan(C1[x1, :])): continue
      x1_curr = np.nanargmin(C1[x1, :])
      curc = C1.copy()
      c1 = curc[x1, x1_curr]
      curc[x1, x1_curr] = np.nan
      curc[np.isnan(curc)] = np.inf
      c2 = np.min(curc[x1, :])
      if c2 / (c1 + 0.0001) < 2:
        C[x1, :ncurr] = maxcost

    # If a next detection has 2 matches then break the tracklet
    for x1 in range(nnext):
      if np.all(np.isnan(C1[:,x1])): continue
      x1_curr = np.nanargmin(C1[:,x1])
      curc = C1.copy()
      c1 = curc[x1_curr,x1]
      curc[x1_curr,x1] = np.nan
      curc[np.isnan(curc)] = np.inf
      c2 = np.min(curc[:,x1])
      if c2/(c1+0.0001) < 2:
        C[:nnext,x1] = maxcost

  # match
  idxcurr, idxnext = opt.linear_sum_assignment(C)

  costs = C[idxcurr, idxnext]
  cost = np.sum(costs)
  
  # idxnext < nnext, idxcurr < ncurr means we are assigning
  # an existing id
  idsnext = -np.ones(nnext, dtype=int)
  isassigned = np.logical_and(idxnext < nnext, idxcurr < ncurr)
  idsnext[idxnext[isassigned]] = idscurr[idxcurr[isassigned]]
  
  # idxnext < nnext, idxcurr >= ncurr means we are creating
  # a new trajectory
  if np.isnan(lastid):
    lastid = np.max(idscurr)
  idxbirth = idxnext[np.logical_and(idxnext < nnext, idxcurr >= ncurr)]
  for i in range(np.size(idxbirth)):
    lastid += 1
    idsnext[idxbirth[i]] = lastid
  
  if params['verbose'] > 1:
    isdeath = np.logical_and(idxnext >= nnext, idxcurr < ncurr)
    logging.info('N. ids assigned: %d, N. births: %d, N. deaths: %d' % (
      np.count_nonzero(isassigned), np.size(idxbirth), np.count_nonzero(isdeath)))
  
  return idsnext, lastid, cost, costs

def assign_ids(trk, params, T=np.inf):
  """
  assign_ids(trk,params)
  Assign identities to each detection in each frame so that one-to-one
  inter-frame match cost is minimized. Matching between frames t and t+1
  is done using match_frame.
  Input:
  trk: Trk object, where Trk.pTrk[:,:,:,t] are the
  detections for frame t. All coordinates will be nan if the number of
  detections in a given frame is less than maxnanimals.
  params: dictionary of parameters (see match_frame for details).
  Output: ids is a Tracklet representation of a maxnanimals x T matrix with
  integers 0, 1, ... indicating the identity of each detection in each frame.
  -1 is assigned to dummy detections.
  """
  
  # p is d x nlandmarks x maxnanimals x T
  # nan is used to indicate missing data
  T = int(np.minimum(T, trk.T))
  T1 = trk.T0+T-1
  pcurr = trk.getframe(trk.T0)
  idxcurr = trk.real_idx(pcurr)
  pcurr = pcurr[:, :, idxcurr]
  ids = TrkFile.Tracklet(defaultval=-1, size=(1, trk.ntargets, T))
  # allocate for speed!
  [sf, ef] = trk.get_startendframes()
  ids.allocate((1,), sf-trk.T0, np.minimum(T-1, ef-trk.T0))
  # ids = -np.ones((trk.T,trk.ntargets),dtype=int)
  idscurr = np.arange(np.count_nonzero(idxcurr), dtype=int)
  
  ids.settargetframe(idscurr, np.where(idxcurr.flatten())[0], 0)
  # ids[idxcurr,0] = idscurr
  lastid = np.max(idscurr)
  costs = np.zeros(T-1)
  
  set_default_params(params)
  
  for t in tqdm(range(trk.T0, T1+1)):
    pnext = trk.getframe(t)
    idxnext = trk.real_idx(pnext)
    pnext = pnext[:, :, idxnext]
    idsnext, lastid, costs[t-1-trk.T0], _ = \
      match_frame(pcurr, pnext, idscurr, params, lastid)
    ids.settargetframe(idsnext, np.where(idxnext.flatten())[0], t-trk.T0)
    # ids[t,idxnext] = idsnext
    pcurr = pnext
    idscurr = idsnext
  return ids, costs

def match_frame_id(pcurr, pnext, idcost, params, defaultval=np.nan):
  """
  match_frame_id(pcurr,pnext,idcost,params,maxcost=None)
  Uses the Hungarian algorithm to match targets tracked in the current
  frame with targets detected in the next frame. The cost of
  assigning target i to detection j is the L1 error between the
  2*nlandmarks dimensional vectors normalized by the number of landmarks.
  The cost of a trajectory birth or death is params['maxcost']/2. Thus,
  it is preferable to kill one trajectory and create another if
  the matching error is > params['maxcost']
  Inputs:
  d x nlandmarks x ncurr positions of landmarks of nnext animals
  detected in the next frame
  idscurr: ncurr array, integer ids of the animals tracked in the
  current frame
  params: dictionary of parameters.
  lastid: (optional) scalar, last id used in tracking so far, if there are
  trajectory births, they will start with id lastid+1
  Outputs:
  idsnext: nnext array, integer ids assigned to animals in next frame
  Parameters:
  params['maxcost']: The cost of a trajectory birth or death is
  params['maxcost']/2. Thus, it is preferable to kill one trajectory
  and create another if the matching error is > params['maxcost'].
  params['verbose']: Whether to print out information
  """
  
  # pcurr: d x nlandmarks x ntargets
  # pnext: d x nlandmarks x nnext
  # nlast: ntargets
  
  # check sizes
  nlandmarks = pcurr.shape[0]
  d = pcurr.shape[1]
  ntargets = pcurr.shape[-1]
  nnext = pnext.shape[-1]
  assert pnext.shape[0] == nlandmarks, \
    'N landmarks do not match, curr = %d, next = %d' % (nlandmarks, pnext.shape[0])
  assert pnext.shape[1] == d, \
    'Dimensions do not match, curr = %d, next = %d' % (d, pnext.shape[1])
  # which ids are assigned in the current frame
  idxcurr = TrkFile.real_idx(pcurr,defaultval).flatten()
  ncurr = np.count_nonzero(idxcurr)
  
  # construct the cost matrix
  # C[i,j] is the cost of matching curr[i] and next[j]
  C = np.zeros((ntargets+nnext, ntargets+nnext))
  # missing prediction
  C[:ntargets,nnext:] = params['cost_missing']
  # extra predictions
  C[ntargets:,:nnext] = params['cost_extra']
  pcurr = np.reshape(pcurr, (d * nlandmarks, ntargets, 1))
  pnext = np.reshape(pnext, (d * nlandmarks, 1, nnext))
  D = np.zeros((ntargets,nnext))
  D[idxcurr,:] = np.sum(np.abs(pcurr[:,idxcurr,:]-pnext), axis=0) / nlandmarks * params['weight_movement']
  Cmovement = C.copy()
  Cmovement[:ntargets, :nnext] = D
  C[:ntargets, :nnext] = D+idcost
  
  # match
  idxcurr, idxnext = opt.linear_sum_assignment(C)
  costs = C[idxcurr, idxnext]
  cost = np.sum(costs)
  
  # idxnext < nnext, idxcurr < ncurr means we are assigning
  # an existing id
  isassigned = np.logical_and(idxnext < nnext, idxcurr < ntargets)
  idsnext = -np.ones(nnext, dtype=int)
  idsnext[idxnext[isassigned]] = idxcurr[isassigned]

  ismissing = np.logical_and(idxnext >= nnext, idxcurr < ntargets)
  isextra = np.logical_and(idxnext < nnext, idxcurr >=ntargets)

  stats = {}
  stats['nmissing'] = np.count_nonzero(ismissing)
  stats['nextra'] = np.count_nonzero(isextra)
  stats['cost_movement'] = np.sum(Cmovement[idxcurr,idxnext])
  stats['cost_id'] = np.sum(idcost[idxcurr[isassigned],idxnext[isassigned]])
  stats['npred'] = nnext
  
  if params['verbose'] > 1:
    print('N. ids assigned: %d, N. extra detections: %d, N. missing detections: %d' % (
      np.count_nonzero(isassigned), stats['extra'], stats['nmissing']))
  
  return idsnext, cost, costs, stats

def assign_recognize_ids(trk, idcosts, params, T=np.inf):
  """
  assign_recognize_ids(trk,idcosts,params,T=inf)
  Assign identities to each detection in each frame so that both the one-to-one
  inter-frame match cost and the individual identity costs are minimized. Matching
  for frame t is done using match_frame_id.
  Input:
  trk: Trk object, where Trk.pTrk[:,:,:,t] are the
  detections for frame t. All coordinates will be nan if the number of
  detections in a given frame is less than maxnanimals.
  idcosts: list of length >= T, where idcosts[t] corresponds to frame t
  and idcosts[t][i,j] is the cost of assigning prediction i to target j.
  params: dictionary of parameters (see match_frame for details).
  T: scalar, number of frames to run assign for
  Output: ids is a Tracklet representation of a maxnanimals x T matrix with
  integers 0, 1, ... indicating the identity of each detection in each frame.
  -1 is assigned to dummy detections.
  """
  
  # p is d x nlandmarks x maxnanimals x T
  # nan is used to indicate missing data
  T = int(np.minimum(T, trk.T))
  T1 = trk.T0+T-1
  pcurr = trk.getframe(trk.T0)
  idxcurr = trk.real_idx(pcurr)
  pcurr = pcurr[:, :, idxcurr]
  ids = TrkFile.Tracklet(defaultval=-1, size=(1, T, trk.ntargets))
  # allocate for speed!
  [sf, ef] = trk.get_startendframes()
  ids.allocate((1,), sf, np.minimum(sf+T-1, ef))
    
  # idcosts is a len T list of ntargets x npreds[t] matrices
  ntargetsreal = idcosts[0].shape[0]
  costs = np.zeros(T)
  
  set_default_params(params)
  
  # save some statistics for debugging
  stats = {'nmissing': np.zeros(T, dtype=int), 'nextra': np.zeros(T, dtype=int), 'cost_movement': np.zeros(T),
           'cost_id': np.zeros(T), 'npred': np.zeros(T)}

  t = trk.T0
  pnext = trk.getframe(t)
  npts = pnext.shape[0]
  d = pnext.shape[1]
  npred = pnext.shape[3]
  pnext = pnext.reshape((npts,d,npred))

  # set ids in first frame based on idcosts only
  # idsnext[i] is which id prediction i was matched to
  idsnext,costs[0],_,statscurr = match_frame_id(np.zeros(pnext.shape),np.zeros(pnext.shape),idcosts[t-trk.T0],params,defaultval=trk.defaultval)
  ids.settargetframe(idsnext, np.where(idxcurr.flatten())[0], t)
  for key in statscurr.keys():
    stats[key][t-trk.T0] = statscurr[key]
    
  # initial nlast -- array storing number of frames since each id was last detected
  nlast = np.zeros(ntargetsreal,dtype=int)
  nlast[:] = params['maxframes_missed']
  pnext = pcurr
  
  for t in tqdm(range(trk.T0+1, T1+1)):
    
    # set pcurr based on pnext and idsnext from previous time point
    pcurr[:,:,idsnext[idsnext>=0]] = pnext[:,:,idsnext>=0]
    isdetected = np.isin(np.arange(ntargetsreal,dtype=int),idsnext)
    nlast += 1
    nlast[isdetected] = 0
    # only set pcurr to nan if it's been a very long time since we last detected this target
    pcurr[:,:,nlast>params['maxframes_missed']] = np.nan

    # read in the next frame positions
    pnext = trk.getframe(t)
    isnext = trk.real_idx(pnext)
    pnext = pnext[:, :, isnext]
    # main matching
    idsnext, costs[t-trk.T0], _, statscurr = \
      match_frame_id(pcurr, pnext, idcosts[t-trk.T0], params,defaultval=trk.defaultval)
    for key in statscurr.keys():
      stats[key][t-trk.T0] = statscurr[key]
    ids.settargetframe(idsnext, np.where(isnext.flatten())[0], t)

  if params['verbose'] > 0:
    print('Frames analyzed: %d, Extra detections: %d, Missed detections: %d'%(T,np.sum(stats['nextra']),np.sum(stats['nmissing'])))
    print('Frames with both extra and missed detections: %d'%(np.count_nonzero(np.logical_and(stats['nextra']>0,stats['nmissing']>0))))
    print('N. predictions: min: %d, mean: %f, max: %d'%(np.min(stats['npred']),np.mean(stats['npred']),np.max(stats['npred'])))
    prctiles_compute = [5.,10.,25.,50.,75.,90.,95.]
    cost_movement_prctiles = np.percentile(stats['cost_movement'],prctiles_compute)
    cost_id_prctiles = np.percentile(stats['cost_id'],prctiles_compute)
    print('Percentiles of movement, id cost:' )
    for i in range(len(prctiles_compute)):
      print('%dth percentile: %f, %f'%(prctiles_compute[i],cost_movement_prctiles[i],cost_id_prctiles[i]))

  return ids, costs, stats


def stitch(trk, ids, params):
  """
  stitch(trk,ids,params): Fill in short gaps (<= params['maxframes_missed']) to
  connect trajectory deaths and births.
  :param trk: Trk class object with detections
  :param ids: Tracklet class object indicating ids assigned to each detection, output of assign_ids
  :param params: parameters dict. Only relevant parameter is 'maxframes_missed'
  :return: ids: Updated identity assignment matrix after stitching
  :return: isdummy: Tracklet class object representing nids x T matrix indicating whether a frame is missed for a given id.
  """
  _, maxv = ids.get_min_max_val()
  nids = np.max(maxv)+1
  # nids = np.max(ids)+1
  
  # get starts and ends for each id
  t0s = np.zeros(nids, dtype=int)
  t1s = np.zeros(nids, dtype=int)
  for id in range(nids):
    idx = ids.where(id)
    # idx = np.nonzero(id==ids)
    t0s[id] = np.min(idx[1])
    t1s[id] = np.max(idx[1])
  
  # isdummy = np.zeros((ids.ntargets,ids.T),dtype=bool)
  isdummy = TrkFile.Tracklet(defaultval=False, size=(1, nids, ids.T))
  isdummy.allocate((1,), t0s, t1s)
  
  allt1s = np.unique(t1s)
  assert allt1s[-1] == ids.T-1
  # skip deaths in last frame
  for i in range(len(allt1s)-1):
    t = allt1s[i]
    # all ids that end this frame
    ids_death = np.nonzero(t1s == t)[0]
    idscurr = ids.getframe(t)
    assert idscurr.shape[0]==1 and idscurr.shape[1]==1, 'Values returned by getframe have shape (1,1,ntgt)'
    if ids_death.size == 0:
      continue
    lastid = np.max(ids_death)
    pcurr = np.zeros((trk.nlandmarks, trk.d, ids_death.size))
    assert np.any(isdummy.gettargetframe(ids_death, t)) == False
    
    for j in range(ids_death.size):
      pcurr[:, :, j] = trk.gettargetframe(np.where(idscurr == ids_death[j])[2], t).reshape((trk.nlandmarks, trk.d))
      # pcurr[:,:,j] = p[:,:,ids[:,t]==ids_death[j],t].reshape((d,nlandmarks))
    for nframes_skip in range(2, params['maxframes_missed']+2):
      # all ids that start at frame t+nframes_skip
      ids_birth = np.nonzero(t0s == t+nframes_skip)[0]
      if ids_birth.size == 0:
        continue
      assert np.any(isdummy.gettargetframe(ids_birth, t+nframes_skip)) == False
      # assert np.any(isdummy[ids_birth,t+nframes_skip])==False
      pnext = np.zeros((trk.nlandmarks, trk.d, ids_birth.size))
      for j in range(ids_birth.size):
        pnext[:, :, j] = trk.gettargetframe(np.where(ids.getframe(t+nframes_skip) == ids_birth[j])[2],
                                            t+nframes_skip).reshape((trk.nlandmarks, trk.d))
        # pnext[:,:,j]=p[:,:,ids[:,t+nframes_skip]==ids_birth[j],t+nframes_skip].reshape((d,nlandmarks))
      # try to match
      maxcost = params['maxcost_missed'][np.minimum(params['maxcost_missed'].size-1, nframes_skip-2)]
      idsnext, _, _, _ = match_frame(pcurr, pnext, ids_death, params, lastid, maxcost=maxcost)
      # idsnext[j] is the id assigned to ids_birth[j]
      ismatch = idsnext <= lastid
      if not np.any(ismatch):
        continue
      for j in range(idsnext.size):
        id_death = idsnext[j]
        if id_death > lastid:
          continue
        id_birth = ids_birth[j]
        ids.replace(id_birth, id_death)
        # ids[ids==id_birth] = id_death
        idx = np.nonzero(ids_death == id_death)
        pcurr = np.delete(pcurr, idx[0], axis=2)
        ids_death = np.delete(ids_death, idx[0])
        t0s[id_birth] = -1
        t1s[id_death] = t1s[id_birth]
        t1s[id_birth] = -1
        isdummy.settargetframe(np.ones((1, nframes_skip-1), dtype=bool), id_death,
                               np.arange(t+1, t+nframes_skip, dtype=int))
        # isdummy[id_death,t+1:t+nframes_skip] = True
        if params['verbose'] > 0:
          logging.info('Stitching id %d frame %d to id %d frame %d' % (id_death, t, id_birth, t+nframes_skip))
      
      if ids_death.size == 0:
        break
  
  return ids, isdummy


def delete_short(ids, isdummy, params):
  """
  delete_short(ids,params):
  Delete trajectories that are at most params['maxframes_delete'] frames long.
  :param ids: maxnanimals x T matrix indicating ids assigned to each detection, output of assign_ids, stitch
  :param isdummy: nids x T matrix indicating whether a frame is missed for a given id.
  :param params: parameters dict. Only relevant parameter is 'maxnframes_delete'
  :return: ids: Updated identity assignment matrix after deleting
  """
  
  _, maxv = ids.get_min_max_val()
  nids = np.max(maxv)+1
  # nids=np.max(ids)+1
  
  # get starts and ends for each id
  t0s = -np.ones(nids, dtype=int)
  t1s = -np.ones(nids, dtype=int)
  nframes = np.zeros(nids, dtype=int)
  for id in range(nids):
    idx = ids.where(id)
    if not np.any(idx[1]):
      continue
    t0s[id] = np.min(idx[1])
    t1s[id] = np.max(idx[1])
    isdummycurr = isdummy.gettargetframe(id, np.arange(t0s[id], t1s[id]+1, dtype=int))
    nframes[id] = np.count_nonzero(isdummycurr == False)
  ids_short = np.nonzero(np.logical_and(nframes <= params['maxframes_delete'], t0s >= 0))[0]
  for id in ids_short:
    ids.replace(id, -1)
  # ids[np.isin(ids,ids_short)] = -1
  if params['verbose'] > 0:
    logging.info('Deleting %d short trajectories' % ids_short.size)
  return ids, ids_short


def delete_lowconf(trk, ids, params):
  """
  delete_lowconf(ids,params):
  Delete trajectories that have mean confidence lower than params['minconf_delete'] frames long.
  :param ids: maxnanimals x T matrix indicating ids assigned to each detection, output of assign_ids, stitch
  :param isdummy: nids x T matrix indicating whether a frame is missed for a given id.
  :param params: parameters dict. Only relevant parameter is 'maxnframes_delete'
  :return: ids: Updated identity assignment matrix after deleting
  """

  _, maxv = ids.get_min_max_val()
  nids = np.max(maxv) + 1
  tot_conf = np.zeros(nids)
  tot_count = np.zeros(nids)
  sf,ef = trk.get_startendframes()

  for tid in range(trk.ntargets):
    _,edict = trk.gettarget(tid,True)
    cur_ids = ids.gettarget(tid)
    assert cur_ids.shape[0]==1 and cur_ids.shape[2] == 1, 'Ids returned should have shape (1,nframes,1)'
    cur_ids = cur_ids[0,:,0][sf[tid]:(ef[tid]+1)]
    cur_conf = edict['pTrkConf'].mean(axis=0)
    cur_conf = cur_conf[(sf[tid]-trk.T0):(ef[tid]+1-trk.T0)]
    for j in range(nids):
      tot_conf[j] += np.nansum(cur_conf[cur_ids==j])
      tot_count[j] += np.nansum(cur_conf[cur_ids==j]>0)
  mean_conf = tot_conf/(tot_count+0.00001)
  ids_lowconf = np.nonzero(mean_conf<params['minconf_delete'])[0]
  for id in ids_lowconf:
    ids.replace(id, -1)
  if params['verbose'] > 0:
    logging.info('Deleting %d trajectories with low confidence' % ids_lowconf.size)
  return ids, ids_lowconf


def merge(trk,ids):
  p_ndx = min(ids)
  trk.pTrk[:, :, :, p_ndx] = np.nanmean(trk.pTrk[...,ids],-1)
  to_remove = ([i for i in ids if i!=p_ndx])

  trk.pTrk = np.delete(trk.pTrk,to_remove,-1)
  for k in trk.trkFields:
    if trk.__dict__[k] is not None:
      trk.__dict__[k] = np.delete(trk.__dict__[k],to_remove,-1)

  trk.ntargets = trk.ntargets-len(to_remove)


def merge_close(trk, params):
  """
  merge_close(trk,params):
  Delete trajectories that have are on average closer than params['maxcost'].
  :param params: parameters dict. Only relevant parameter is 'maxcost'
  """

  rm_count = 0
  orig_count = trk.ntargets
  while True:
    dist_trk = np.nanmean(np.abs(trk.pTrk[...,None,:]-trk.pTrk[...,None]).sum(1).mean(0),axis=0)
    dist_trk[np.diag_indices(dist_trk.shape[0])] = np.inf
    id1,id2 = np.unravel_index(np.nanargmin(dist_trk), dist_trk.shape)
    if dist_trk[id1,id2]>params['maxcost']:
      break
    merge(trk,[id1,id2])
    rm_count +=1

  logging.info(f'Removing {rm_count} out of {orig_count} trajectories by merging them into other trajectories that are close')



def estimate_maxcost(trk, nsample=1000, prctile=95., mult=None, nframes_skip=1, heuristic='secondorder'):
  """
  maxcost = estimate_maxcost(trk,nsample=1000,prctile=95.,mult=None,nframes_skip=1,heuristic='secondorder')
  Estimate the threshold for the maximum cost for matching identities. This is done
  by running match_frame on some sample frames, looking at the assignment costs
  assuming all assignments are allowed, and then taking a statistic of all those
  assignment costs.
  The heuristic used is maxcost = 2.* mult .* percentile(allcosts,prctile)
  where prctile and mult are parameters
  :param trk: Trk object
  :param nsample: Number of frames to sample, default = 1000
  :param prctile: Percentile used when computing threshold, default = 95.
  :param mult: Multiplier used when computing threshold , default = 100./prctile
  :param nframes_skip: Number of frames to skip, default = 1
  :param heuristic: How to convert statistics of costs to a threshold.
  Options: 'secondorder' (Mayank's heuristic), 'prctile' (Kristin's heuristic).
  Default: 'secondorder'.
  Returns threshold on cost.
  """
  
  if (mult is None) and heuristic =='prctile':
    mult = 100. / prctile
  elif mult is None:
    mult = 1.2
  nsample = np.minimum(trk.T, nsample)
  tsample = np.round(np.linspace(trk.T0, trk.T1-nframes_skip-1, nsample)).astype(int)
  params = {}
  minv, maxv = trk.get_min_max_val()
  minv = np.min(minv, axis=0)
  maxv = np.max(maxv, axis=0)
  bignumber = np.sum(maxv-minv) * 2.1
  # bignumber = np.sum(np.nanmax(p,axis=(1,2,3))-np.nanmin(p,axis=(1,2,3)))*2.1
  params['maxcost'] = bignumber
  params['verbose'] = 0
  set_default_params(params)
  allcosts = np.zeros((trk.ntargets, nsample))
  allcosts[:] = np.nan

  for i in range(nsample):
    t = tsample[i]
    pcurr = trk.getframe(t)
    pnext = trk.getframe(t+nframes_skip)
    pcurr = pcurr[:, :, trk.real_idx(pcurr)]
    pnext = pnext[:, :, trk.real_idx(pnext)]
    if (pcurr.size<1) or (pnext.size<1): continue
    ntargets_curr = pcurr.shape[2]
    ntargets_next = pnext.shape[2]
    idscurr = np.arange(ntargets_curr)
    idsnext, _, _, costscurr = match_frame(pcurr, pnext, idscurr, params,force_match=True)
    ismatch = np.isin(idscurr, idsnext)
    assert np.count_nonzero(ismatch) == np.minimum(ntargets_curr, ntargets_next)
    costscurr = costscurr[:ntargets_curr]
    allcosts[:np.count_nonzero(ismatch), i] = costscurr[ismatch]
  
  isdata = np.isnan(allcosts) == False
  
  if heuristic == 'prctile':
    maxcost = mult * np.percentile(allcosts[isdata], prctile)
  elif heuristic == 'secondorder':
    # use sharp increase in 2nd order differences.
    qq = np.percentile(allcosts[isdata], np.arange(50, 100, 0.25))
    dd1 = qq[1:] - qq[:-1]
    dd2 = dd1[1:] - dd1[:-1]
    all_ix = np.where(dd2 > 4)[0]
    # threshold is where the second order increases by 4, so sort of the coefficient for the quadratic term.
    if len(all_ix) < 1:
        ix = 198 # choose 98 % as backup
    else:
        ix = all_ix[0]
    ix = np.clip(ix,5,198)
    logging.info('nframes_skip = %d, choosing %f percentile of link costs with a value of %f to decide the maxcost'%(nframes_skip,ix/4+50,qq[ix]))
    maxcost = mult*qq[ix]
  
  return maxcost
  
  # debug code -- what are the differences between having no threshold on cost and having the chosen threshold
  # params['maxcost'] = maxcost
  #
  # for i in range(nsample):
  #     t=tsample[i]
  #     pcurr=p[:,:,:,t]
  #     pnext=p[:,:,:,t+1]
  #     pcurr=pcurr[:,:,real_idx(pcurr)]
  #     pnext=pnext[:,:,real_idx(pnext)]
  #     ntargets_curr=pcurr.shape[2]
  #     ntargets_next=pnext.shape[2]
  #     idscurr=np.arange(ntargets_curr)
  #     idsnext,_,_,costscurr=match_frame(pcurr,pnext,idscurr,params)
  #     ismatch=np.isin(idscurr,idsnext)
  #     nmiss = np.minimum(ntargets_curr,ntargets_next) - np.count_nonzero(ismatch)
  #     if nmiss > 0:
  #         sortedcosts = -np.sort(-allcosts[:,i])
  #         logging.info('i = %d, t = %d, nmiss = %d, ncurr = %d, nnext = %d, costs removed: %s'%(i,t,nmiss,ntargets_curr,ntargets_next,str(sortedcosts[:nmiss])))


def estimate_maxcost_missed(trk, maxframes_missed, nsample=1000, prctile=95., mult=None, heuristic='secondorder'):
  """
  maxcost_missed = estimate_maxcost_missed(trk,maxframes_missednsample=1000,prctile=95.,mult=None, heuristic='secondorder')
  Estimate the threshold for the maximum cost for matching identities across > 1 frame.
  This is done by running match_frame on some sample frames, looking at the assignment costs assuming all assignments
  are allowed, and then taking a statistic of all those assignment costs.
  The heuristic used is maxcost = 2.* mult .* percentile(allcosts,prctile)
  where prctile and mult are parameters.
  :param trk: Trk object
  :param maxframes_missed: How many frames can be skipped
  :param nsample: Number of frames to sample
  :param prctile: Percentile used when computing threshold
  :param mult: Multiplier used when computing threshold
  :param heuristic: How to convert statistics of costs to a threshold.
  Options: 'secondorder' (Mayank's heuristic), 'prctile' (Kristin's heuristic).
  Default: 'secondorder'.
  Returns np.ndarray containing threshold on cost for each number of frames missed.
  """
  
  maxcost_missed = np.zeros(maxframes_missed)
  for nframes_skip in range(2, maxframes_missed+2):
    maxcost_missed[nframes_skip-2] = estimate_maxcost(trk, prctile=prctile, mult=mult, nframes_skip=nframes_skip, nsample=nsample,heuristic=heuristic)
  return maxcost_missed


def set_default_params(params):
  if 'verbose' not in params:
    params['verbose'] = 1
  if 'weight_movement' not in params:
    params['weight_movement'] = 1.
  if 'maxframes_missed' not in params:
    params['maxframes_missed'] = np.inf


def test_assign_ids():
  """
  test_assign_ids():
  constructs some synthetic data and makes sure assign_ids works
  """
  
  # random.seed(2)
  d = 2
  nlandmarks = 17
  n0 = 6
  minn = 3
  pbirth = .5
  pdeath = .5
  T = 20
  maxnbirthdeath = 2
  
  params = {}
  params['maxcost'] = .1
  # params['verbose'] = 1
  
  # create some data
  p = np.zeros((nlandmarks, d, T, n0))
  p[:] = np.nan
  ids = -np.ones((T, n0))
  
  pcurr = random.rand(nlandmarks, d, n0)
  p[:, :, 0, :] = pcurr
  idscurr = np.arange(n0)
  ids[0, :] = idscurr
  lastid = np.max(idscurr)
  
  for t in range(1, T):
    
    idxcurr = TrkFile.real_idx(pcurr,np.nan)
    ncurr = np.count_nonzero(idxcurr)
    pnext = pcurr[:, :, idxcurr]
    idsnext = idscurr
    for i in range(maxnbirthdeath):
      if ncurr > minn and random.rand(1) <= pdeath:
        pnext = pnext[:, :, :-1]
        idsnext = idsnext[:-1]
        logging.info('%d: death' % t)
    for i in range(maxnbirthdeath):
      if random.rand(1) <= pbirth:
        lastid += 1
        pnext = np.concatenate((pnext, random.rand(nlandmarks, d, 1)), axis=2)
        idsnext = np.append(idsnext, lastid)
        logging.info('%d: birth' % t)
    nnext = pnext.shape[2]
    if nnext > p.shape[3]:
      pad = np.zeros((nlandmarks, d, T, nnext-p.shape[3]))
      pad[:] = np.nan
      p = np.concatenate((p, pad), axis=3)
      ids = np.concatenate((ids, -np.ones((T, nnext-ids.shape[1]))), axis=1)
    perm = random.permutation(nnext)
    pnext = pnext[:, :, perm]
    idsnext = idsnext[perm]
    p[:, :, t, :nnext] = pnext
    ids[t, :nnext] = idsnext
    
    pcurr = pnext
    idscurr = idsnext
  
  logging.info('ids = ')
  logging.info(str(ids))
  ids1, costs = assign_ids(TrkFile.Trk(p=p), params)
  
  logging.info('assigned ids = ')
  logging.info(str(ids1))
  logging.info('costs = ')
  logging.info(str(costs))
  
  issameid = np.zeros((ids.shape[0]-1, ids.shape[1]**2))
  for t in range(ids.shape[0]-1):
    issameid[t, :] = (ids[t, :].reshape((ids.shape[1], 1)) == ids[t+1, :].reshape((1, ids.shape[1]))).flatten()
  
  ids1d = ids1.getdense()
  ids1d = ids1d.reshape((ids1d.shape[1:]))
  issameid1 = np.zeros((ids1d.shape[0]-1, ids1d.shape[1]**2))
  for t in range(ids1d.shape[0]-1):
    issameid1[t, :] = (ids1d[t, :].reshape((ids1d.shape[1], 1)) == ids1d[t+1, :].reshape((1, ids1d.shape[1]))).flatten()
  
  assert np.all(issameid1 == issameid)


def test_match_frame():
  """
  test_match_frame():
  constructs some synthetic data and makes sure match_frame works
  """
  
  d = 2
  nlandmarks = 17
  ncurr = 6
  nnext = ncurr+1
  
  pcurr = random.rand(d, nlandmarks, ncurr)
  pnext = np.zeros((d, nlandmarks, nnext))
  if nnext < ncurr:
    pnext = pcurr[:, :, :nnext]
  else:
    pnext[:, :, :ncurr] = pcurr
    pnext[:, :, ncurr:] = random.rand(d, nlandmarks, nnext-ncurr)
  
  idscurr = np.arange(0, ncurr)
  lastid = np.max(idscurr)
  
  perm = random.permutation(nnext)
  pnext = pnext[:, :, perm]
  
  params = {}
  params['maxcost'] = .8
  params['verbose'] = 1
  
  idsnext, lastid, cost, _ = match_frame(pcurr, pnext, idscurr, params, lastid)
  logging.info('permutation = '+str(perm))
  logging.info('idsnext = '+str(idsnext))
  logging.info('cost = %f' % cost)


def mixed_colormap(n, cmfun=cm.jet):
  idx0 = np.linspace(0., 1., n)
  cm0 = cmfun(idx0)
  
  d = np.abs(idx0.reshape((1, n))-idx0.reshape((n, 1)))
  idx = np.zeros(n, dtype=int)
  mind = d[0, :]
  mind[0] = -np.inf
  for i in range(1, n):
    j = np.argmax(mind)
    idx[i] = j
    mind = np.minimum(mind, d[j, :])
    mind[j] = -np.inf
  cm1 = cm0[idx, :]
  return cm1

def nonmax_supp(trk, params):
  for t in range(trk.T0,trk.T1+1):
    pcurr = trk.getframe(t)
    curd = np.abs(pcurr[...,0,:,None]-pcurr[...,0,None,:]).sum(1).mean(0)
    curd[np.diag_indices(curd.shape[0])] = np.inf
    if np.all(np.isnan(curd)|np.isinf(curd)): continue
    id1,id2 = np.where(curd<params['nms_max'])
    groups = []
    for ndx in range(len(id1)):
      done = False
      for g in groups:
        if g.count(id1[ndx])>0:
          done = True
          if g.count(id2[ndx])==0:
            g.append(id2[ndx])
        if g.count(id2[ndx])>0:
          done = True
          if g.count(id1[ndx])==0:
            g.append(id1[ndx])
      if not done:
        groups.append([id1[ndx],id2[ndx]])

    for g in groups:
      p_ndx = g[0]
      to_remove = g[1:]
      pcurr[...,0,p_ndx] = np.mean(trk.pTrk[:,:,t,g],axis=2)
      pcurr[...,0,to_remove] = np.nan
      trk.setframe(pcurr,t)

def link_trklets(trk,conf,mov,out_file):
  if conf.multi_stitch_id:
    conf1 = copy.deepcopy(conf)
    conf1.imsz = conf1.multi_stitch_id_cropsz

    if len(conf1.ht_pts)>0:
      conf1.use_ht_trx = True
      conf1.trx_align_theta = True
    else:
      conf1.use_bbox_trx = True
      conf1.trx_align_theta = False
    return link_id(trk,mov,conf1,out_file)
  else:
    return link(trk,conf)


def get_default_params(conf):
  # Update some of the parameters based on conf
  params = {}
  params['verbose'] = 1
  params['maxframes_missed'] = 10
  params['maxframes_delete'] = 10
  params['maxcost_prctile'] = 95.
  params['maxcost_mult'] = 3
  params['maxcost_framesfit'] = 3
  params['maxcost_heuristic'] = 'prctile'
  params['minconf_delete'] = 0.5
  params['nms_prctile'] = 50
  return params


def link(trk,conf,params_in=None,do_merge_close=False,do_stitch=True,do_delete_short=False):
  # pred_locs is nfr x nanimals x npts x 2

  params = get_default_params(conf)
  if params_in != None:
    params.update(params_in)
  nframes_test = np.inf

  # locs_lnk = np.transpose(pred_locs, [2, 3, 0, 1])
  # if pred_conf is None:
  #   locs_conf = None
  # else:
  #   locs_conf = np.transpose(pred_conf,[2,0,1])
  # if pred_animal_conf is None:
  #   locs_animal_conf = None
  # else:
  #   locs_animal_conf = np.transpose(pred_animal_conf,[2,0,1])
  # ts = np.ones_like(locs_lnk[:,0, ...]) * apt.datetime2matlabdn()
  # tag = np.zeros(ts.shape).astype('bool')  # tag which is always false for now.
  # trk = TrkFile.Trk(p=locs_lnk, pTrkTS=ts, pTrkTag=tag,pTrkConf=locs_conf,pTrkAnimalConf=locs_animal_conf)

  T = np.minimum(np.inf, trk.T)
  # p should be d x nlandmarks x maxnanimals x T, while pTrk is nlandmarks x d x T x maxnanimals
  # p = np.transpose(trk['pTrk'],(1,0,3,2))
  nframes_test = int(np.minimum(T, nframes_test))
  if 'maxcost' not in params:
    params['maxcost'] = estimate_maxcost(trk, prctile=params['maxcost_prctile'], mult=params['maxcost_mult'], heuristic=params['maxcost_heuristic'])
  if 'maxcost_missed' not in params:
    params['maxcost_missed'] = estimate_maxcost_missed(trk, params['maxcost_framesfit'], prctile=params['maxcost_prctile'], mult=params['maxcost_mult'], heuristic=params['maxcost_heuristic'])
  if 'nms_max' not in params:
    params['nms_max'] = estimate_maxcost(trk, prctile=params['nms_prctile'], mult=1, heuristic='prctile')

  logging.info('maxcost set to %f' % params['maxcost'])
  logging.info('maxcost_missed set to ' + str(params['maxcost_missed']))
  nonmax_supp(trk, params)
  ids, costs = assign_ids(trk, params, T=nframes_test)
  if isinstance(ids, np.ndarray):
    nids_original = np.max(ids) + 1
  else:
    _, nids_original = ids.get_min_max_val()
    nids_original = nids_original + 1

  if do_stitch:
    ids, isdummy = stitch(trk, ids, params)
  else:
    _, maxv = ids.get_min_max_val()
    nids = np.max(maxv) + 1
    # nids = np.max(ids)+1

    # get starts and ends for each id
    t0s = np.zeros(nids, dtype=int)
    t1s = np.zeros(nids, dtype=int)
    for id in range(nids):
      idx = ids.where(id)
      # idx = np.nonzero(id==ids)
      t0s[id] = np.min(idx[1])
      t1s[id] = np.max(idx[1])

    # isdummy = np.zeros((ids.ntargets,ids.T),dtype=bool)
    isdummy = TrkFile.Tracklet(defaultval=False, size=(1, nids, ids.T))
    isdummy.allocate((1,), t0s, t1s)

  if do_delete_short:
    ids, ids_short = delete_short(ids, isdummy, params)
#  if locs_conf is not None:
#    ids,ids_lowconf = delete_lowconf(trk,ids,params)
  _, ids = ids.unique()
  trk.apply_ids(ids)
  if do_merge_close:
    merge_close(trk,params)
  return trk


def link_id(trk,mov_file,conf,out_file, params_in=None):
  params = {}
  params['maxframes_delete']:3
  if params_in is not None:
    params.udpate(params_in)
  trk = link(trk,conf, params_in=params,do_merge_close=False,do_stitch=False)
  train_data, tmp_trx  = get_id_train_images(trk,mov_file,conf)
  id_classifier, loss_history = train_id_classifier(train_data,conf, trk)
  wt_out_file = out_file.replace('.trk','_idwts.p')
  torch.save({'model_state_params':id_classifier.state_dict()},wt_out_file)

  def_params = get_default_params(conf)
  trk_out, matched = link_trklet_id(trk,id_classifier,mov_file,conf,tmp_trx,min_len=def_params['maxframes_delete'])
  return trk_out


def get_id_train_images(trk_in,mov_file,conf,n_ex=10000,batch_size=8,num_workers=12):
  ss, ee = trk_in.get_startendframes()
  tmp_trx = tempfile.mkstemp()[1]
  trk_in.save(tmp_trx,saveformat='tracklet')
  # Save the current trk to be used as trx. Could be avoided but the whole image patch extracting pipeline exists with saved trx file, so not rewriting it.

  num_done = 0
  min_trx_len = 100
  if np.count_nonzero((ee-ss)>min_trx_len)<conf.max_n_animals:
    min_trx_len = np.percentile((ee-ss),20)-1

  cap = movies.Movie(mov_file)
  trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
  trx = trx_dict['trx']
  cap.close()

  sel_trk = np.where((ee - ss) > min_trx_len)[0]
  sel_trk_info = list(zip(sel_trk, ss[sel_trk], ee[sel_trk]))

  data = read_ims_par(trx, sel_trk_info, mov_file, conf)

  # logging.info('Creating ID data generator ...')
  # id_dat = id_dset(mov_file,trx,conf,to_do_list)
  # id_loader = torch.utils.data.DataLoader(id_dat,batch_size=batch_size,num_workers=num_workers,worker_init_fn=lambda id: np.random.seed(id),pin_memory=True)
  # id_iter = iter(id_loader)

  return data, tmp_trx

def get_overlap(ss_t,ee_t,ss,ee, curidx):
  # For overlap either the start of the trajectory should lie within the range or the end
  # Since trk ends go to last frame + 1, less and greater comparisons have to be done carefully
  starts = np.maximum(ss_t,ss)
  ends = np.minimum(ee_t,ee)
  overlap_amt = np.array([len(range(st,en))/(ee-ss) for st,en in zip(starts,ends)])
  overlap_tgts = np.where(overlap_amt>0)[0]
  overlap_tgts = np.array(list(set(overlap_tgts) - set([curidx])))
  overlap_amt = overlap_amt[overlap_tgts]

  # overlaps = ((ss_t >= ss) & (ss_t <  ee)) | \
  #            ((ee_t >  ss) & (ee_t <= ee)) | \
  #            ((ss >= ss_t) & (ss <  ee_t)) | \
  #            ((ee >  ss_t) & (ee <= ee_t))
  # overlap_tgts = np.where(overlaps)[0]
  # overlap_tgts = np.array(list(set(overlap_tgts) - set([curidx])))
  return overlap_tgts, overlap_amt


class id_dset(torch.utils.data.IterableDataset):

  def __init__(self, data, t_dist, self_dist, overlap_dist, ss_t, ee_t, t_preds, n_tr, confd, rescale, distort=True):
      self.all_data = [data, t_dist, self_dist,overlap_dist,ss_t,ee_t,t_preds,n_tr,confd,rescale,distort]

  def __iter__(self):
    [data, t_dist, self_dist, overlap_dist, ss_t, ee_t, t_preds, n_tr, confd,rescale, distort] = self.all_data
    while True:
      curims = []
      while len(curims) < 1:

        if np.random.rand() < 0.5:
          self_dist1 = self_dist+0.2
          sample_wt = self_dist1 / self_dist1.sum()
        else:
          sample_wt = 2.2 - np.clip(overlap_dist, 0, 2)
          sample_wt = sample_wt / sample_wt.sum()

        curidx = np.random.choice(n_tr, p=sample_wt)
        cur_dat = data[curidx]
        overlap_tgts, overlap_amt = get_overlap(ss_t,ee_t,ss_t[curidx], ee_t[curidx], curidx)
        if overlap_tgts.size < 1: continue
        if t_preds is None:
          t_dist_all = np.ones([len(overlap_tgts),cur_dat[0].shape[0], cur_dat[0].shape[0]])
          t_dist_self = np.ones([cur_dat[0].shape[0], cur_dat[0].shape[0]])
        else:
          t_dist_all = t_dist[curidx,overlap_tgts]
          t_dist_self = t_dist[curidx,curidx]

        wt_self = (t_dist_self + 0.2).sum(axis=1)
        wt_self = wt_self / wt_self.sum()
        idx_self1 = np.random.choice(len(cur_dat[0]), p=wt_self)
        im1 = cur_dat[0][idx_self1]
        wt_self2 = t_dist_self[idx_self1] + 0.2
        wt_self2 = wt_self2 / wt_self2.sum()
        idx_self2 = np.random.choice(len(cur_dat[0]), p=wt_self2)
        im2 = cur_dat[0][idx_self2]

        t_dist_overlap_idx = (t_dist_all[:,idx_self1] + t_dist_all[:,idx_self2]) / 2
        overlap_wts = 2.2 - np.clip(t_dist_overlap_idx, 0, 2)
        overlap_wts = overlap_wts*overlap_amt[:,None]
        o_sh = overlap_wts.shape
        overlap_wts = overlap_wts.flatten()
        overlap_sel = np.random.choice(len(overlap_wts), p=overlap_wts / overlap_wts.sum())
        overlap_tgt_ndx, overlap_im_idx = np.unravel_index(overlap_sel, o_sh)
        overlap_tgt = overlap_tgts[overlap_tgt_ndx]

        overlap_im = data[overlap_tgt][0][overlap_im_idx]
        curims.append(np.stack([im1, im2, overlap_im], 0))

      curims = np.array(curims)
      curims = curims.reshape((-1,) + curims.shape[2:])
      curims = process_id_ims(curims, confd, distort, rescale)
      curims = curims.astype('float32')
      yield curims


def process_id_ims(curims, conf, distort, rescale):
  if curims.shape[3] == 1:
    curims = np.tile(curims, [1, 1, 1, 3])
  dummy_locs = np.ones([curims.shape[0],2,2]) * curims.shape[1]/2
  zz, _ = PoseTools.preprocess_ims(curims, dummy_locs, conf, distort, rescale)
  zz = zz.transpose([0, 3, 1, 2])
  zz = zz / 255.
  im_mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
  im_std = np.array([[[0.229]], [[0.224]], [[0.225]]])
  zz = zz - im_mean
  zz = zz / im_std
  return zz

def read_ims_par(trx, trk_info, mov_file, conf,n_ex=50):
  n_threads = min(24, mp.cpu_count())
  with mp.get_context('spawn').Pool(n_threads) as pool:

    data_files = pool.starmap(read_tracklet_ims, [(trx, trk_info[n::n_threads], mov_file, conf, n_ex, np.random.randint(100000)) for n in range(n_threads)])

  data = []
  for curf in data_files:
    data.append(PoseTools.pickle_load(curf))
    os.remove(curf)

  data = [i for sublist in data for i in sublist]
  return data


def read_tracklet_ims(trx, trk_info, mov_file, conf, n_ex,seed):
  np.random.seed(seed)
  cap = movies.Movie(mov_file)

  all_ims = []
  for cur_trk in trk_info:
    rand_frs = []
    while len(rand_frs) < n_ex:
      cur_fr = np.random.choice(np.arange(cur_trk[1], cur_trk[2]+1))
      if np.isnan(trx[cur_trk[0]]['x'][0,cur_fr-cur_trk[1]]):
        continue
      rand_frs.append(cur_fr)

    cur_list = [[fr, cur_trk[0]] for fr in rand_frs]

    ims = apt.create_batch_ims(cur_list, conf, cap, False, trx, None, use_bsize=False)
    all_ims.append([ims, cur_trk[0],cur_trk[1],cur_trk[2],cur_list])

  tfile = tempfile.mkstemp()[1]
  with open(tfile,'wb') as f:
    pickle.dump(all_ims,f)
  cap.close()
  return tfile


def tracklet_pred(ims, net, conf, rescale):
    preds = []
    for ix in range(len(ims)):
        curims = ims[ix]
        zz = process_id_ims(curims, conf, False, rescale)
        zz = zz.astype('float32')
        zz = torch.tensor(zz).cuda()
        with torch.no_grad():
            oo = net(zz).cpu().numpy()
        preds.append(oo)

    rr = np.array(preds)
    return rr

def train_id_classifier(data, conf, trk, n_iters=40000,save=False,save_file=None,rescale=1,flip90=False, bsz=16, use_sampling=True, num_times_sample=10,debug=False):

  class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
      super(ContrastiveLoss, self).__init__()
      self.margin = margin

    def forward(self, output1, output2, label):
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
      loss_contrastive = torch.sum((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(
        torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive

  loss_history = []
  net = models.resnet.resnet18(pretrained=True)
  net.fc = torch.nn.Linear(in_features=512, out_features=32, bias=True)

  net = net.cuda()
  criterion = ContrastiveLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.0001)

  confd = copy.deepcopy(conf)
  if confd.trx_align_theta:
    confd.rrange = 10.
  else:
    confd.rrange = 180.
  confd.trange = min(conf.imsz) / 15
  confd.horzFlip = False
  confd.vertFlip = False
  confd.scale_factor_range = 1.
  confd.brange = [-0.05, 0.05]
  confd.crange = [0.95, 1.05]

  logging.info('Training ID network ...')
  net.train()
  net = net.cuda()

  ss, ee = trk.get_startendframes()
  tgt_id = np.array([r[1] for r in data])
  ss_t = ss[tgt_id]
  ee_t = ee[tgt_id]
  t_preds = None
  t_dist = None
  n_tr = len(data)
  self_dist = np.ones(n_tr)
  overlap_dist = np.ones(n_tr)
  sampling_period = round(n_iters/num_times_sample)

  train_dset = id_dset(data, t_dist, self_dist, overlap_dist, ss_t, ee_t, t_preds, n_tr, confd, rescale, True)
  n_workers = 10 if not debug else 0
  train_loader = torch.utils.data.DataLoader(train_dset, batch_size=bsz, pin_memory=True, num_workers=n_workers,worker_init_fn=lambda id: np.random.seed(id))
  train_iter = iter(train_loader)

  for epoch in tqdm(range(n_iters)):

    if epoch % sampling_period == 0 and epoch > 0 and use_sampling:
      net = net.eval()
      ims = [dd[0] for dd in data]
      t_preds = tracklet_pred(ims, net, confd, rescale)
      t_dist = np.ones([n_tr,n_tr,t_preds.shape[1],t_preds.shape[1]])*np.nan
      overlap_dist = np.ones(n_tr)*np.nan
      for xx in range(n_tr):
        cur_dist =  np.linalg.norm(t_preds[xx:xx+1,:, None] - t_preds[:, None], axis=-1)
        t_dist[xx] = cur_dist
        overlap_tgts, overlap_amt = get_overlap(ss_t,ee_t,ss_t[xx], ee_t[xx], xx)
        if overlap_tgts.size < 1: continue
        overlap_dist[xx] = np.mean(cur_dist[overlap_tgts]*overlap_amt[:,None,None])

      self_dist = np.mean(t_dist[range(n_tr),range(n_tr)],axis=(1,2))

      del train_iter, train_loader, train_dset
      train_dset =  id_dset(data,t_dist, self_dist,overlap_dist,ss_t,ee_t,t_preds,n_tr,confd,rescale,True)
      train_loader = torch.utils.data.DataLoader(train_dset,batch_size=bsz,pin_memory=True,num_workers=10,worker_init_fn=lambda id: np.random.seed(id))
      train_iter = iter(train_loader)

      net = net.train()

    curims = next(train_iter).cuda()
    curims = curims.reshape((-1,)+ curims.shape[2:])
    optimizer.zero_grad()
    output = net(curims)
    output = output.reshape((-1,3) + output.shape[1:])
    output1, output2, output3 = output[:,0], output[:,1], output[:,2]
    l1 = criterion(output1, output2, 0)
    l2 = criterion(output1, output3, 1)
    l3 = criterion(output2, output3, 1)
    loss_contrastive = l1 + l2 + l3
    loss_contrastive.backward()
    optimizer.step()
    if epoch%5000==0 and save and save_file is not None and epoch>0:
      wt_out_file = f'{save_file}-{epoch}.p'
      torch.save({'model_state_params': net.state_dict(),'loss_history':loss_history}, wt_out_file)

    loss_history.append(loss_contrastive.item())

##
  wt_out_file = f'{save_file}-{n_iters}.p'
  torch.save({'model_state_params': net.state_dict(), 'loss_history': loss_history}, wt_out_file)
  del train_iter, train_loader, train_dset
  return net, loss_history


def link_trklet_id(trk, net, mov_file, conf, tmp_trx, n_per_trk=50,min_len=0,rescale=1, min_len_select=5):
  cap = movies.Movie(mov_file)
  ss, ee = trk.get_startendframes()
  trx_dict = apt.get_trx_info(tmp_trx, conf, cap.get_n_frames())
  trx = trx_dict['trx']

  # For each tracklet chose n_per_trk random examples and the find their embedding.
  sel_tgt = np.where((ee-ss+1)>=min_len_select)[0]
  sel_ss = ss[sel_tgt]; sel_ee = ee[sel_tgt]
  trk_info = list(zip(sel_tgt, sel_ss, sel_ee))
  logging.info(f'Sampling images from {len(sel_ss)} tracklets to assign identity to the tracklets ...')
  start_t = time.time()
  data = read_ims_par(trx, trk_info, mov_file, conf, n_ex=n_per_trk)
  end_t = time.time()
  logging.info(f'Sampling images took {round((end_t-start_t)/60)} minutes')
  tgt_id = np.array([r[1] for r in data])

  net.eval()
  preds = []
  for ix in tqdm(sel_tgt):
    curndx = np.where(tgt_id==ix)[0][0]
    curims = data[curndx][0]
    zz = process_id_ims(curims, conf, False, rescale)
    zz = torch.Tensor(zz).float().cuda()
    with torch.no_grad():
      oo = net(zz).cpu().numpy()
    preds.append(oo)

  cap.close()

  # Now stitch the tracklets based on their identity
  trk.convert2dense()

  rr = np.array(preds)
  thres = 1.
  assigned_ids = np.ones(trk.ntargets) * -1
  to_remove = list(set(range(trk.ntargets))-set(sel_tgt))

  logging.info('Stitching tracklets based on identity ...')
  # for xx in range(rr.shape[0]):
  #   cur_tgt = sel_tgt[xx]
  #   if cur_tgt in to_remove:
  #     continue
  #   dd = np.linalg.norm(rr[xx, None, ...] - rr[:, :, None], axis=-1)
  #   dd = dd.reshape([dd.shape[0], -1])
  #   dd = np.percentile(dd, 50, axis=-1)
  #   # Find distance between all the patches of current tracklet to other tracklets. Match the tracklets if at least 70% of distances are less than the threshold
  #
  #   close_ids = np.where(dd < thres)[0]
  #   len_trk = sel_ee[close_ids] - sel_ss[close_ids]
  #   close_ids = close_ids[np.argsort(-len_trk)]
  #   for cid in close_ids:
  #     match_tgt = sel_tgt[cid]
  #     if cid <= xx:        continue
  #     if match_tgt in to_remove:
  #       print(f'Not joining trajectory {match_tgt} to {cur_tgt} (dist={dd[cid]}) because it was joined to {[assigned_ids[match_tgt]]}')
  #     elif (ss[cur_tgt] <= ss[match_tgt] <= ee[cur_tgt]) or (ss[cur_tgt] <= ee[match_tgt] <= ee[cur_tgt]):
  #       # Don't join if there is an overlap. If Id classifier is good, this shouldn't happen, but can happen in cases when the two animals overlap heavily and the ID classifier doesn't know what to do
  #       print(f'Range doesnt match for {cur_tgt} and {match_tgt} (dist={dd[cid]}). Not joining')
  #     else:
  #       trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, cur_tgt] = trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, match_tgt]
  #       to_remove.append(match_tgt)
  #       assigned_ids[match_tgt] = cur_tgt

  groups = cluster_tracklets_id(rr,sel_ss,sel_ee)
  for gr in groups:
    cur_tgt = min(sel_tgt[gr])

    for gg in gr:
      if sel_tgt[gg] == cur_tgt: continue
      match_tgt = sel_tgt[gg]
      trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, cur_tgt] = trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, match_tgt]
      to_remove.append(match_tgt)
      assigned_ids[match_tgt] = cur_tgt

  # Delete the trks that have been merged
  trk.pTrk = np.delete(trk.pTrk, to_remove, -1)
  for k in trk.trkFields:
    if trk.__dict__[k] is not None:
      trk.__dict__[k] = np.delete(trk.__dict__[k], to_remove, -1)
  trk.ntargets = trk.ntargets - len(to_remove)


  logging.info(f'Deleting short trajectories with length less than {min_len}')
  # delete short tracks
  sf,ef = trk.get_startendframes()
  to_remove = np.where( (ef-sf)<=min_len)[0]
  # Delete the trks that have been merged
  trk.pTrk = np.delete(trk.pTrk, to_remove, -1)
  for k in trk.trkFields:
    if trk.__dict__[k] is not None:
      trk.__dict__[k] = np.delete(trk.__dict__[k], to_remove, -1)
  trk.ntargets = trk.ntargets - len(to_remove)

  trk.convert2sparse()
  return trk, assigned_ids


def cluster_tracklets_id(embed,sel_ss, sel_ee):

  n_tr = embed.shape[0]
  n_ex = embed.shape[1]
  ddr = np.ones([n_tr, n_tr, n_ex, n_ex]) * np.nan

  for xx in tqdm(range(n_tr)):
    ddr[xx, :] = np.linalg.norm(embed[xx, None, :, None] - embed[:, None, :], axis=-1)
  ddm = np.median(ddr, axis=(2, 3))
  # plt.figure(); plt.imshow(ddm)

  import scipy.spatial.distance as ssd
  from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

  ddm[range(n_tr), range(n_tr)] = 0.
  distArray = ssd.squareform( ddm)
  Z = linkage(distArray, 'average')
  # plt.figure()
  # dn = dendrogram(Z)

  ##
  thres = 1.
  F = fcluster(Z, thres, criterion='distance')

  groups = []
  n_fr = max(sel_ee)
  tr_len = sel_ee - sel_ss
  for ndx in range(max(F)):
    cur_gr = np.where(np.array(F) == (ndx + 1))[0]
    cur_gr_ord = np.argsort(-tr_len[cur_gr])
    cur_gr = cur_gr[cur_gr_ord]

    cur_groups = [[],]
    ctline = np.zeros(n_fr)
    for cc in cur_gr:
      prev_overlap = np.sum(ctline[sel_ss[cc]:sel_ee[cc]])/(sel_ee[cc]-sel_ss[cc])
      if prev_overlap>0.05:
        cur_groups.append([cc])
      else:
        cur_groups[0].append(cc)
        ctline[sel_ss[cc]:sel_ee[cc]] +=1

    groups.extend(cur_groups)

  return groups


def test_assign_ids_data():
  """
  test_assign_ids_data:
  loads data from a trkfile and runs assign_ids, stitch, delete_short, and unique on them
  :return:
  """
  
  matplotlib.use('TkAgg')
  plt.ion()
  
  trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2.trk.part'
  outtrkfile = '/groups/branson/bransonlab/apt/tmp/200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2_kbstitched_tracklet.trk'
  
  #trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1.trk.part'
  #outtrkfile = '/groups/branson/bransonlab/apt/tmp/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1_kbstitched_v2.trk'
  
  # parameters
  params = {}
  params['verbose'] = 1
  params['maxframes_missed'] = 10
  params['maxframes_delete'] = 10
  params['maxcost_prctile'] = 95.
  params['maxcost_mult'] = 1.25
  params['maxcost_framesfit'] = 3
  params['maxcost_heuristic'] = 'secondorder'
  nframes_test = np.inf
  
  showanimation = False
  
  trk = TrkFile.Trk(trkfile)
  T = np.minimum(np.inf, trk.T)
  # p should be d x nlandmarks x maxnanimals x T, while pTrk is nlandmarks x d x T x maxnanimals
  # p = np.transpose(trk['pTrk'],(1,0,3,2))
  nframes_test = int(np.minimum(T, nframes_test))
  params['maxcost'] = estimate_maxcost(trk, prctile=params['maxcost_prctile'], mult=params['maxcost_mult'],heuristic=params['maxcost_heuristic'])
  params['maxcost_missed'] = estimate_maxcost_missed(trk, params['maxcost_framesfit'],prctile=params['maxcost_prctile'], mult=params['maxcost_mult'],heuristic=params['maxcost_heuristic'])
  logging.info('maxcost set to %f' % params['maxcost'])
  logging.info('maxcost_missed set to ' + str(params['maxcost_missed']))
  ids, costs = assign_ids(trk, params, T=nframes_test)
  if isinstance(ids, np.ndarray):
    nids_original = np.max(ids)+1
  else:
    _, nids_original = ids.get_min_max_val()
    nids_original = nids_original+1
  
  ids, isdummy = stitch(trk, ids, params)
  ids, ids_short = delete_short(ids, isdummy, params)
  _, ids = ids.unique()
  trk.apply_ids(ids)
  
  # save to file
  trk.save(outtrkfile)
  # TrkFile.save_trk(outtrkfile,newtrk)
  
  plt.figure()
  nids = trk.ntargets
  # nids = newtrk['pTrk'].shape[3]
  logging.info('%d ids in %d frames, removed %d ids' % (nids, nframes_test, nids_original-nids))
  nidsplot = int(np.minimum(nids, np.inf))
  minp, maxp = trk.get_min_max_val()
  minp = np.min(minp)
  maxp = np.max(maxp)
  startframes, endframes = trk.get_startendframes()
  
  hax = []
  for d in range(trk.d):
    hax.append(plt.subplot(1, trk.d, d+1))
    hax[d].set_title('coord %d' % d)
  
  for id in range(nidsplot):
    
    logging.info('Target %d, %d frames (%d to %d)' % (id, endframes[id]-startframes[id]+1, startframes[id], endframes[id]))
    
    ts = np.arange(startframes[id], endframes[id]+1, dtype=int)
    n = ts.size
    p = trk.gettargetframe(id, ts).reshape((trk.nlandmarks, trk.d, n))
    mu = np.nanmean(p, axis=0)
    idxnan = np.where(np.all(np.isnan(mu), axis=0))[0]
    for d in range(trk.d):
      h, = hax[d].plot(ts, mu[d, :], '.-')
      if d == 0:
        color = h.get_color()
      hax[d].plot(ts[0], mu[d, 0], 'o', color=color, mfc=color)
      hax[d].plot(ts[-1], mu[d, -1], 's', color=color, mfc=color)
      if idxnan.size > 0:
        hax[d].plot(ts[idxnan], np.zeros(idxnan.size), 'x', color=color)
  plt.show(block=True)
  
  if showanimation:
    
    colors = mixed_colormap(nids)
    colors[:, :4] *= .75
    plt.figure()
    h = [None, ] * nids
    htrail = [None, ] * nids
    hax = plt.gca()
    hax.set_ylim((minp, maxp))
    hax.set_xlim((minp, maxp))
    traillen = 50
    trail = np.zeros((trk.d, traillen, trk.ntargets))
    trail[:] = np.nan
    plt.show(block=False)
    
    T0 = np.nanmin(startframes)
    for t in range(T0, np.nanmax(endframes)+1):
      p = trk.getframe(t)
      isrealidx = trk.real_idx(p).flatten()
      mu = np.nanmean(p, axis=0).reshape((trk.d, trk.ntargets))
      off = t-T0
      if off < traillen:
        trail[:, off, :] = mu
      else:
        trail = np.append(trail[:, 1:, :], mu.reshape((trk.d, 1, nids)), axis=1)
      for id in range(nids):
        if t > endframes[id] or t < startframes[id]:
          if htrail[id] is not None:
            htrail[id].remove()
            htrail[id] = None
        else:
          if htrail[id] is None:
            htrail[id], = plt.plot(trail[0, :, id], trail[1, :, id], '-', color=colors[id, :] * .5+np.ones(4) * .5)
          else:
            htrail[id].set_data(trail[0, :, id], trail[1, :, id])
      
      for id in np.where(isrealidx)[0]:
        if h[id] is None:
          h[id], = plt.plot(p[:, 0, :, id].flatten(), p[:, 1, :, id].flatten(), '.-', color=colors[id, :])
        else:
          h[id].set_data(p[:, 0, :, id].flatten(), p[:, 1, :, id].flatten())
      for id in np.where(~isrealidx)[0]:
        if h[id] is not None:
          h[id].remove()
          h[id] = None
      plt.pause(.01)


def test_estimate_maxcost():
  
  matplotlib.use('TkAgg')
  plt.ion()
  
  trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1.trk.part'
  
  # parameters
  params = {}
  params['verbose'] = 1
  params['maxframes_missed'] = 10
  params['maxframes_delete'] = 10
  params['maxcost_prctile'] = 95.
  params['maxcost_mult'] = 1.25
  params['maxcost_framesfit'] = 3
  
  trk = TrkFile.Trk(trkfile=trkfile)
  # frames should be consecutive
  # assert np.all(np.diff(trk['pTrkFrm'], axis=1) == 1), 'pTrkFrm should be consecutive frames'
  # p should be d x nlandmarks x maxnanimals x T, while pTrk is nlandmarks x d x T x maxnanimals
  # p = np.transpose(trk['pTrk'], (1, 0, 3, 2))
  
  maxcost0 = estimate_maxcost(trk, prctile=params['maxcost_prctile'], mult=params['maxcost_mult'])
  maxcost1 = estimate_maxcost_missed(trk, params['maxcost_framesfit'],
                                     prctile=params['maxcost_prctile'], mult=params['maxcost_mult'])
  maxcost = np.append(np.atleast_1d(maxcost0), maxcost1.flatten())
  
  plt.figure()
  plt.plot(np.arange(maxcost.size)+1, maxcost, 'o-')
  plt.show(block=True)
  
def test_recognize_ids():
  
  matplotlib.use('tkAgg')
  plt.ion()
  
  # locations of data
  expdir = '/groups/branson/bransonlab/apt/experiments/data/200918_m170234vocpb_m170234_odor_m170232_f0180322_allframes'
  rawtrkfile = os.path.join(expdir,'rawtrk.trk')
  outtrkfile = os.path.join(expdir,'kbstiched_debug.trk')

  trxfile = os.path.join(expdir,'trx.mat')
  dell2ellfile = os.path.join(expdir,'perframe/dell2ell.mat')
  moviefile = os.path.join(expdir,'movie.mjpg')
  movieidxfile = os.path.join(expdir,'index.txt')

  # landmarks to use for calculating a centroid to compare to motr tracking
  bodylandmarks = np.array([0,1])
  plotlandmarkorder = np.array([0,2,3,0,1])

  # max ell2ell distance for motr tracking to be considered close
  distthresh = 10
  # fill holes of size < close_diameter
  close_diameter = 5
  # whether to use a normalized version of idcosts such that total costs sum to 1 for each prediction
  # I think this makes more sense, but it makes setting parameters harder...
  normalize_idcosts = True

  # debugging - how many frames to test
  nframes_test = np.inf

  # big number, means don't assign here
  BIGCOST = 100000.
  
  # whether to plot an animation of tracking results
  showanimation = False

  # whether to plot
  plot_debug_input = False
  
  # parameters for matching
  params = {}
  # printing debugging output
  params['verbose'] = 1
  # weight of the movement cost (weight of j cost is 1)
  if normalize_idcosts:
    params['weight_movement'] = 1./100.
  else:
    params['weight_movement'] = 1.
  # cost of setting a target to not be detected
  params['cost_missing'] = 50.*params['weight_movement']
  # cost of having a prediction that is not assigned to a target
  params['cost_extra'] = 50.*params['weight_movement']
  # if a target is not detected, we use its last know location for matching.
  # if it's been more than maxframes_missed frames since last location, then
  # don't use this past location in assigning ids
  params['maxframes_missed'] = 20
  
  # load in unlinked data
  trk = TrkFile.Trk(trkfile=rawtrkfile)
  if not trk.issparse:
    trk.convert2sparse()
  npts = trk.size[0]
  d = trk.size[1]
  
  # load in motr tracking
  trx = TrkFile.load_trx(trxfile)
  # load in pre-computed dell2ell data
  dell2ell = TrkFile.load_perframedata(dell2ellfile)

  # frames tracked by APT
  T0,T1 = trk.get_frame_range()
  T1 = int(np.minimum(T1,T0+nframes_test-1))
  T = T1-T0+1
  ntargets = trk.ntargets

  # there are 2 targets, so we only need one of dell2ell
  assert len(trx['x']) == 2
  dell2ell = dell2ell[0][T0-trx['startframes'][0]:T1-trx['startframes'][0]+1]
  isclose = dell2ell <= distthresh
  if close_diameter > 0:
    se_close = np.ones(close_diameter)
    isclose = scipy.ndimage.morphology.binary_closing(isclose,se_close)

  # APT issue where same prediction returned twice sometimes -- remove duplicates
  ndup = 0
  for t in range(T0,T1+1):
    x = trk.getframe(t)
    D = np.sum(np.abs(x.reshape((npts*d,1,ntargets))-x.reshape((npts*d,ntargets,1))),axis=0)
    D[np.tril_indices(ntargets,k=0)] = np.inf
    (i,j) = np.where(D<=.1)
    if i.size > 0:
      x[:,:,:,j] = trk.defaultval
      trk.setframe(x,t)
      ndup += 1
  print('%d duplicate values found and removed'%ndup)

  if plot_debug_input:
    # plot trx and trk info to make sure they line up in time
    plt.figure()
    t = T0
    p = trk.getframe(t)
    for i in range(trk.ntargets):
      plt.plot(p[:,0,0,i],p[:,1,0,i],'r.')
      if (t >= trx['startframes'][i]) and (t <= trx['endframes'][i]):
        plt.plot(trx['x'][i][t-trx['startframes'][i]],trx['y'][i][t-trx['startframes'][i]],'o')
    ax = plt.gca()
    ax.set_aspect('equal')

    # plot dell2ell
    plt.figure()
    plt.plot(np.where(isclose)[0]-trx['startframes'][0]+T0,dell2ell[isclose],'.')
    plt.plot(np.where(~isclose)[0]-trx['startframes'][0]+T0,dell2ell[~isclose],'.')
    plt.legend(['close','far'])
    plt.xlabel('Frame')
    plt.ylabel('ell2ell distance (mm)')
  
  # compute motr-based assignment costs
  idcosts = [None,]*T
  for t in range(T0,T1+1):
    i = t - T0
    pcurr = trk.getframe(t)
    idxreal = trk.real_idx(pcurr)
    pcurr = pcurr[:,:,idxreal]
    npred = pcurr.shape[2]
    if isclose[i]:
      idcosts[i] = np.zeros((ntargets,npred))
      continue
    center = np.reshape(np.mean(pcurr[bodylandmarks,:,:],axis=0),[2,1,npred])
    trxpos = np.zeros((2,ntargets,1))
    trxpos[:] = np.nan
    ntrxcurr = 0
    for j in range(ntargets):
      if (t >= trx['startframes'][j]) and (t <= trx['endframes'][j]):
        trxpos[0, j, 0] = trx['x'][j][t-trx['startframes'][j]]
        trxpos[1, j, 0] = trx['y'][j][t-trx['startframes'][j]]
        ntrxcurr+=1
    D = np.sqrt(np.sum(np.square(center-trxpos),axis=0)) # ntargets x npred
    badidx = np.isnan(D)
    if normalize_idcosts:
      z = np.nansum(D,axis=0)
      z[z<=0.] = 1.
      D = D/z
    D[badidx] = BIGCOST
    idcosts[i] = D

  if plot_debug_input:
    maxnpred = trk.ntargets
    y = np.zeros((T,maxnpred))
    y[:] = np.nan
    for t in range(T0,T1+1):
      i = t - T0
      y[i,:idcosts[i].shape[1]] = idcosts[i][0,:]-idcosts[i][1,:]
  
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
    for i in range(maxnpred):
      ax[0].plot(y[:,i],'.',label='Prediction %d'%i)
    
    for i in range(len(trx['x'])):
      t0 = np.maximum(T0,trx['startframes'][i])
      t1 = np.minimum(T1,trx['endframes'][i])
      ax[1].plot(np.arange(t0-T0,t1+1-T0),trx['x'][i][t0-trx['startframes'][i]:t1+1-trx['startframes'][i]],'x',label='Motr target %d'%i)
      ax[2].plot(np.arange(t0-T0,t1+1-T0),trx['y'][i][t0-trx['startframes'][i]:t1+1-trx['startframes'][i]],'x',label='Motr target %d'%i)
    
    for i in range(maxnpred):
      p = trk.gettargetframe(i,np.arange(T0,T1+1,dtype=int))
      center = np.mean(p[bodylandmarks,:,:,:],axis=0)
      ax[1].plot(center[0,:,0],'.',label='Prediction %d'%i)
      ax[2].plot(center[1,:,0],'.',label='Prediction %d'%i)
  
      
    ax[0].title.set_text('j cost difference')
    ax[1].title.set_text('x-coordinate')
    ax[2].title.set_text('y-coordinate')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
  
  # compute id assignment
  nframes_test = int(np.minimum(T, nframes_test))
  ids, costs, stats = assign_recognize_ids(trk, idcosts, params, T=nframes_test)
  
  # apply to tracking
  trk.apply_ids(ids)

  # save linked tracking to output
  trk.save(outtrkfile)

  # plot some visualization of the results
  idxextra = np.where(stats['nextra']>0)[0]
  idxmissing = np.where(stats['nmissing']>0)[0]
  idxboth = np.where(np.logical_and(stats['nmissing']>0,stats['nextra']>0))[0]
  
  trk0 = TrkFile.Trk(trkfile=rawtrkfile)
  if not trk0.issparse:
    trk0.convert2sparse()
  
  fig, ax = plt.subplots(trk.ntargets+1, 1, sharex=True, sharey=False)
  ax[0].plot(costs,'.',label='Total cost',zorder=10)
  ax[0].plot(stats['cost_movement'],'.',label='Movement cost',zorder=20)
  ax[0].plot(np.where(~isclose)[0],stats['cost_id'][~isclose],'.',label='Id cost, not close',zorder=30)
  ax[0].title.set_text('Cost')

  for i in range(trk0.ntargets):
    t0 = np.maximum(T0,trx['startframes'][i])
    t1 = np.minimum(T1,trx['endframes'][i])
    p0 = trk0.gettargetframe(i,np.arange(t0,t1+1,dtype=int))
    center0 = np.mean(p0[bodylandmarks,:,:,:],axis=0)
    ax[1].plot(center0[0,:,0],'o',label='Raw %d'%i,zorder=5)
    ax[2].plot(center0[1,:,0],'o',label='Raw %d'%i,zorder=5)
  
  for i in range(trk.ntargets):
    t0 = np.maximum(T0,trx['startframes'][i])
    t1 = np.minimum(T1,trx['endframes'][i])
    p = trk.gettargetframe(i,np.arange(t0,t1+1,dtype=int))
    center = np.mean(p[bodylandmarks,:,:,:],axis=0)
    trxx = trx['x'][i][t0-trx['startframes'][i]:t1+1-trx['startframes'][i]]
    trxy = trx['y'][i][t0-trx['startframes'][i]:t1+1-trx['startframes'][i]]
    if T <= 2000: #plotting slow
      ax[1].plot(np.tile(np.arange(t0-T0,t1+1-T0).reshape(1,t1-t0+1),(trk.ntargets,1)),np.concatenate((trxx.reshape((1,t1-t0+1)),center[0,:,:].reshape(1,t1-t0+1)),axis=0),'k.-')
      ax[2].plot(np.tile(np.arange(t0-T0,t1+1-T0).reshape(1,t1-t0+1),(trk.ntargets,1)),np.concatenate((trxy.reshape((1,t1-t0+1)),center[1,:,:].reshape(1,t1-t0+1)),axis=0),'k.-')
    ax[1].plot(np.arange(t0-T0,t1+1-T0),trxx,'+-',label='Motr %d'%i,zorder=10)
    ax[2].plot(np.arange(t0-T0,t1+1-T0),trxy,'+-',label='Motr %d'%i,zorder=10)

    ax[1].plot(center[0,:,0],'.-',label='Linked %d'%i,zorder=20)
    ax[2].plot(center[1,:,0],'.-',label='Linked %d'%i,zorder=20)
  
  ax[1].title.set_text('x-coordinate')
  ax[2].title.set_text('y-coordinate')
  for i in range(3):
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    #if i > 0:
    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ylim = np.array(ax[i].get_ylim()).reshape((2,1))
    ax[i].plot(np.tile(idxextra,(2,1)),np.tile(ylim,(1,idxextra.size)),'c-',zorder=0)
    ax[i].plot(np.tile(idxmissing,(2,1)),np.tile(ylim,(1,idxmissing.size)),'m-',zorder=0)
    ax[i].plot(np.tile(idxboth,(2,1)),np.tile(ylim,(1,idxboth.size)),'k-',zorder=0)
    ax[i].set_ylim(ylim)

  
  plt.show(block=True)
  if showanimation:
    minp, maxp = trk.get_min_max_val()
    minp = np.min(minp)
    maxp = np.max(maxp)
    
    colors = mixed_colormap(ntargets)
    colors[:, :4] *= .75
    plt.figure()
    h = [None, ] * ntargets
    htrail = [None, ] * ntargets
    hax = plt.gca()
    hax.set_ylim((minp, maxp))
    hax.set_xlim((minp, maxp))
    traillen = 50
    trail = np.zeros((trk.d, traillen, trk.ntargets))
    trail[:] = np.nan
    sf,ef = trk.get_startendframes()
    
    for t in range(T0, T1+1):
      p = trk.getframe(t)
      isrealidx = trk.real_idx(p).flatten()
      mu = np.nanmean(p, axis=0).reshape((trk.d, trk.ntargets))
      off = t-T0
      if off < traillen:
        trail[:, off, :] = mu
      else:
        trail = np.append(trail[:, 1:, :], mu.reshape((trk.d, 1, ntargets)), axis=1)
      for j in range(ntargets):
        if t > ef[j] or t < sf[j]:
          if htrail[j] is not None:
            htrail[j].remove()
            htrail[j] = None
        else:
          if htrail[j] is None:
            htrail[j], = plt.plot(trail[0, :, j], trail[1, :, j], '-', color=colors[j, :] * .5+np.ones(4) * .5)
          else:
            htrail[j].set_data(trail[0, :, j], trail[1, :, j])
      
      for j in np.where(isrealidx)[0]:
        if h[j] is None:
          h[j], = plt.plot(p[plotlandmarkorder, 0, :, j].flatten(), p[plotlandmarkorder, 1, :, j].flatten(), '.-', color=colors[j, :])
        else:
          h[j].set_data(p[plotlandmarkorder, 0, :, j].flatten(), p[plotlandmarkorder, 1, :, j].flatten())
      for j in np.where(~isrealidx)[0]:
        if h[j] is not None:
          h[j].remove()
          h[j] = None
      hax.title.set_text('Frame %d'%t)
      plt.pause(.001)
  
  
  print('finished')
  

if __name__ == '__main__':
  # test_match_frame()
  # test_assign_ids_data()
  test_recognize_ids()
  # test_estimate_maxcost()
  # test_assign_ids()
