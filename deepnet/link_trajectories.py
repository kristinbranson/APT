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
import hdf5storage

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

  strict_match_thres = params['strict_match_thres']

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
      if c2 / (c1 + 0.0001) < strict_match_thres:
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
      if c2/(c1+0.0001) < strict_match_thres:
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

def dummy_ids(trk):
  T = int(trk.T)
  ids = TrkFile.Tracklet(defaultval=-1, size=(1, trk.ntargets, T))
  # allocate for speed!
  [sf, ef] = trk.get_startendframes()
  ids.allocate((1,), sf - trk.T0, np.minimum(T - 1, ef - trk.T0))
  for t in range(trk.ntargets):
    curid = np.ones(ef[t]-sf[t]+1)*t
    ids.settargetframe(curid, t, np.arange(sf[t]-trk.T0,ef[t]-trk.T0+1))
  return ids


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
      pcurr[:, :, j] = trk.gettargetframe(np.where(idscurr == ids_death[j])[2], t+trk.T0).reshape((trk.nlandmarks, trk.d))
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
                                            t+nframes_skip+trk.T0).reshape((trk.nlandmarks, trk.d))
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


def estimate_maxcost(trks, params, params_in=None, nsample=1000, nframes_skip=1):
  if type(trks) not in [list,tuple]:
    trks = [trks]
  if params_in is not None:
    params.update(params_in)

  allcosts = []
  for trk in trks:
    allcosts.append(estimate_maxcost_ind(trk, params, nsample=nsample, nframes_skip=nframes_skip))
  allcosts = np.concatenate(allcosts,axis=0)

  mult = params['maxcost_mult']
  heuristic = params['maxcost_heuristic']
  prctile = params['maxcost_prctile']
  secondorder_thresh = params['maxcost_secondorder_thresh']

  if mult is None:
    if heuristic =='prctile':
      mult = 100. / prctile
    else:
      mult = 1.2

  if heuristic == 'prctile':
    maxcost = mult * np.percentile(allcosts, prctile)
  elif heuristic == 'secondorder':
    # use sharp increase in 2nd order differences.
    isz = 4.
    qq = np.percentile(allcosts, np.arange(50, 100, 1 / isz))
    dd1 = qq[1:] - qq[:-1]
    dd2 = dd1[1:] - dd1[:-1]
    all_ix = np.where(dd2 > secondorder_thresh)[0]
    # threshold is where the second order increases by 4, so sort of the coefficient for the quadratic term.
    if len(all_ix) < 1:
      ix = 198  # choose 98 % as backup
    else:
      ix = all_ix[0]
    ix = np.clip(ix, 5, 198) + 1
    logging.info('nframes_skip = %d, choosing %f percentile of link costs with a value of %f to decide the maxcost' % (
    nframes_skip, ix / isz + 50, qq[ix]))
    maxcost = mult * qq[ix]

  return maxcost


def estimate_maxcost_ind(trk, params, nsample=1000, nframes_skip=1):
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

  nsample = np.minimum(trk.T, nsample)
  tsample = np.round(np.linspace(trk.T0, trk.T1-nframes_skip-1, nsample)).astype(int)
  minv, maxv = trk.get_min_max_val()
  minv = np.min(minv, axis=0)
  maxv = np.max(maxv, axis=0)
  bignumber = np.sum(maxv-minv) * 2.1
  # bignumber = np.sum(np.nanmax(p,axis=(1,2,3))-np.nanmin(p,axis=(1,2,3)))*2.1
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
    idsnext, _, _, costscurr = match_frame(pcurr, pnext, idscurr, params,force_match=True, maxcost=bignumber)
    ismatch = np.isin(idscurr, idsnext)
    assert np.count_nonzero(ismatch) == np.minimum(ntargets_curr, ntargets_next)
    costscurr = costscurr[:ntargets_curr]
    allcosts[:np.count_nonzero(ismatch), i] = costscurr[ismatch]
  
  isdata = np.isnan(allcosts) == False

  return allcosts[isdata]

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


def estimate_maxcost_missed(trk, params, nsample=1000):
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

  maxframes_missed = params['maxcost_framesfit']
  maxcost_missed = np.zeros(maxframes_missed)
  for nframes_skip in range(2, maxframes_missed+2):
    maxcost_missed[nframes_skip-2] = estimate_maxcost(trk, params,  nframes_skip=nframes_skip, nsample=nsample)
  return maxcost_missed


def set_default_params(params):
  if 'verbose' not in params:
    params['verbose'] = 1
  if 'weight_movement' not in params:
    params['weight_movement'] = 1.
  if 'maxframes_missed' not in params:
    params['maxframes_missed'] = np.inf


def get_default_params(conf):
  # Update some of the parameters based on conf
  params = {}
  params['verbose'] = 1
  params['maxframes_missed'] = conf.link_maxframes_missed
  params['maxframes_delete'] = conf.link_maxframes_delete
  params['maxcost_prctile'] = conf.link_maxcost_prctile
  params['maxcost_mult'] = conf.link_maxcost_mult
  params['maxcost_framesfit'] = conf.link_maxcost_framesfit
  params['maxcost_heuristic'] = conf.link_maxcost_heuristic
  params['maxcost_secondorder_thresh'] = conf.link_maxcost_secondorder_thresh
  params['minconf_delete'] = 0.5
  params['strict_match_thres'] = conf.link_strict_match_thres
  return params


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


def link_pure(trk, conf, do_delete_short=False):
  params = get_default_params(conf)

  if 'maxcost' not in params:
    params['maxcost'] = estimate_maxcost(trk, params)
  logging.info('maxcost set to %f' % params['maxcost'])

  if 'maxcost_missed' not in params:
    params['maxcost_missed'] = estimate_maxcost_missed(trk, params)
    logging.info('maxcost_missed set to ' + str(params['maxcost_missed']))

  params['maxframes_delete'] = conf.link_id_min_tracklet_len

  T = np.minimum(np.inf, trk.T)
  nframes_test = np.inf
  nframes_test = int(np.minimum(T, nframes_test))

  ids, costs = assign_ids(trk, params, T=nframes_test)

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
  return trk

  return l_trk

def link_trklets(trk_files, conf, movs, out_files):

  in_trks = [TrkFile.Trk(tt) for tt in trk_files]
  params = get_default_params(conf)

  if 'maxcost' not in params:
    params['maxcost'] = estimate_maxcost(in_trks, params)
  logging.info('maxcost set to %f' % params['maxcost'])

  if 'maxcost_missed' not in params:
    params['maxcost_missed'] = estimate_maxcost_missed(in_trks, params)
    logging.info('maxcost_missed set to ' + str(params['maxcost_missed']))

  params['maxframes_delete'] = conf.link_id_min_tracklet_len

  # if 'nms_max' not in params:
  # params['nms_max'] = estimate_maxcost(trk, prctile=params['nms_prctile'], mult=1, heuristic='prctile')

  #  nonmax_supp(trk, params)

  if conf.link_id:
    conf1 = copy.deepcopy(conf)
    ww = conf1.multi_animal_crop_sz
    conf1.imsz = [ww,ww]

    if len(conf1.ht_pts)>0:
      conf1.use_ht_trx = True
      conf1.trx_align_theta = True
    else:
      conf1.use_bbox_trx = True
      conf1.trx_align_theta = False
    return link_id(in_trks, trk_files, movs, conf1, out_files)

  else:
    out_trks = [link(trk,params) for trk in in_trks]
    return out_trks


def link(trk,params,do_merge_close=False,do_stitch=True,do_delete_short=False):

  ids = dummy_ids(trk)

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


def link_id(trks, trk_files, mov_files, conf, out_files):

  all_trx = []

  for trk_file, mov_file in zip(trk_files,mov_files):
    # l_trk = link(trk, params,do_merge_close=False,do_stitch=False)
    # linked_trks.append(l_trk)

    # Read the linked trk as trx
    # tmp_trk = tempfile.mkstemp()[1]
    # trk.save(tmp_trk,saveformat='tracklet')
    # Save the current trk to be used as trx. Could be avoided but the whole image patch extracting pipeline exists with saved trx file, so not rewriting it.

    cap = movies.Movie(mov_file)
    trx_dict = apt.get_trx_info(trk_file, conf, cap.get_n_frames())
    trx = trx_dict['trx']
    all_trx.append(trx)
    cap.close()

  # train the identity model
  train_data = get_id_train_images(trks, all_trx, mov_files, conf)
  wt_out_file = out_files[0].replace('.trk','_idwts.p')
  id_classifier, loss_history = train_id_classifier(train_data,conf, trks, save_file=wt_out_file)

  # link using idt
  def_params = get_default_params(conf)
  trk_out = link_trklet_id(trks,id_classifier,mov_files,conf, all_trx,min_len_select=def_params['maxframes_delete'])
  return trk_out


def get_id_train_images(linked_trks, all_trx, mov_files, conf):
  all_data = []
  for trk, trx, mov_file in zip(linked_trks,all_trx,mov_files):
    ss, ee = trk.get_startendframes()

    min_trx_len = conf.link_id_min_train_track_len
    if np.count_nonzero((ee-ss+1)>min_trx_len)<conf.max_n_animals:
      min_trx_len = min(1,np.percentile((ee-ss+1),20)-1)

    sel_trk = np.where((ee - ss+1) > min_trx_len)[0]
    sel_trk_info = list(zip(sel_trk, ss[sel_trk], ee[sel_trk]))

    data = read_ims_par(trx, sel_trk_info, mov_file, conf)
    all_data.append(data)
  return all_data

def get_overlap(ss_t,ee_t,ss,ee, curidx):
  # For overlap either the start of the trajectory should lie within the range or the end
  # Since trk ends go to last frame + 1, less and greater comparisons have to be done carefully
  starts = np.maximum(ss_t,ss)
  ends = np.minimum(ee_t+1,ee+1)
  overlap_amt = np.array([len(range(st,en))/(ee-ss+1) for st,en in zip(starts,ends)])
  overlap_tgts = np.where(overlap_amt>0)[0]
  overlap_tgts = np.array(list(set(overlap_tgts) - set([curidx])))

  if overlap_tgts.size == 0:
    overlap_amt = np.array([])
  else:
    overlap_amt = overlap_amt[overlap_tgts]

  # overlaps = ((ss_t >= ss) & (ss_t <  ee)) | \
  #            ((ee_t >  ss) & (ee_t <= ee)) | \
  #            ((ss >= ss_t) & (ss <  ee_t)) | \
  #            ((ee >  ss_t) & (ee <= ee_t))
  # overlap_tgts = np.where(overlaps)[0]
  # overlap_tgts = np.array(list(set(overlap_tgts) - set([curidx])))
  return overlap_tgts, overlap_amt


class id_dset(torch.utils.data.IterableDataset):

  def __init__(self, all_data, mining_dists, trk_data, confd, rescale, valid, distort=True, debug=False):
      self.all_data = [all_data, mining_dists, trk_data, confd, rescale, valid, distort]
      self.debug = debug

  def __iter__(self):
    [all_data, mining_dists, trk_data, confd, rescale, valid, distort] = self.all_data
    while True:
      curims = []
      sel_ndx = np.random.randint(len(all_data))
      data = all_data[sel_ndx]
      dists, overlap_dist_mean, self_dist_mean = mining_dists[sel_ndx]
      ss_t, ee_t, _ = trk_data[sel_ndx]
      n_tr = len(data)

      while len(curims) < 1:

        if np.random.rand() < 0.5:
          self_dist1 = self_dist_mean+0.2
          sample_wt = self_dist1 / self_dist1.sum()
        else:
          sample_wt = 2.2 - np.clip(overlap_dist_mean, 0, 2)
          sample_wt = sample_wt / sample_wt.sum()

        curidx = np.random.choice(n_tr, p=sample_wt)
        cur_dat = data[curidx]

        if not valid:
          overlap_tgts, overlap_amt = get_overlap(ss_t,ee_t,ss_t[curidx],ee_t[curidx],curidx)
          t_dist_all = np.ones([len(overlap_tgts),cur_dat[0].shape[0], cur_dat[0].shape[0]])
          t_dist_self = np.ones([cur_dat[0].shape[0], cur_dat[0].shape[0]])
        else:
          overlap_tgts, overlap_amt = dists[curidx][4:6]
          t_dist_self = dists[curidx][0]
          t_dist_all = dists[curidx][1]

        if overlap_tgts.size < 1: continue

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

        # Do an overlap check
        check = np.zeros(cur_dat[3]-cur_dat[2]+1)
        odata = data[overlap_tgt]
        over_sf = np.maximum(0,odata[2]-cur_dat[2])
        over_ef = np.minimum(cur_dat[3]-cur_dat[2]+1, odata[3]-cur_dat[2]+1)
        check[over_sf:over_ef] = 1
        if check.sum()<1:
          logging.info(f'mov:{sel_ndx}, tr1:{cur_dat[1]}:{cur_dat[2]}-{cur_dat[3]} im1:{cur_dat[4][idx_self1][0]} im2:{cur_dat[4][idx_self2][0]} d:{t_dist_self[idx_self1,idx_self2]}, neg:{odata[1]}:{odata[2]}-{odata[3]}, im3:{odata[4][overlap_im_idx][0]}')
          assert False, 'neg tracklet does not overlap'
        # if self.debug:
        #   logging.info(f'mov:{sel_ndx}, tr1:{cur_dat[1]}:{cur_dat[2]}-{cur_dat[3]} im1:{cur_dat[4][idx_self1][0]} im2:{cur_dat[4][idx_self2][0]} d:{t_dist_self[idx_self1,idx_self2]}, neg:{odata[1]}:{odata[2]}-{odata[3]}, im3:{odata[4][overlap_im_idx][0]}')

        curims.append(np.stack([im1, im2, overlap_im], 0))

      curims = np.array(curims)
      curims = curims.reshape((-1,) + curims.shape[2:])
      curims = process_id_ims(curims, confd, distort, rescale)
      curims = curims.astype('float32')
      yield curims

def process_id_ims_par(im_arr,conf,distort,rescale):
  res_arr = []
  for ims in im_arr:
    res_arr.append(process_id_ims(ims,conf,distort,rescale))
  return res_arr

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
  n_tr = len(trk_info)
  with mp.get_context('spawn').Pool(n_threads) as pool:

    trk_info_split = split_parallel(trk_info,n_threads)
    data_files = pool.starmap(read_tracklet_ims, [(trx, trk_info_split[n], mov_file, conf, n_ex, np.random.randint(100000)) for n in range(n_threads)])

  data = []
  for curf in data_files:
    data.append(PoseTools.pickle_load(curf))
    os.remove(curf)
  data = merge_parallel(data)
  # ndx = np.argsort(np.array([d[1] for d in data]))
  # data = [data[i] for i in ndx]
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

def split_parallel(x,n_threads):
  nx = len(x)
  split = [range((nx * n) // n_threads, (nx * (n + 1)) // n_threads) for n in range(n_threads)]
  split_x = tuple( tuple(x[s] for s in split[n]) for n in range(n_threads))
  assert sum([len(curx) for curx in split_x]) == nx, 'Splitting failed'
  return split_x

def merge_parallel(data):
  data = [i for sublist in data for i in sublist]
  return data

def tracklet_pred(ims, net, conf, rescale):
    preds = []
    n_threads = min(24, mp.cpu_count())
    n_batches = max(1,len(ims)//(3*n_threads))
    n_tr = len(ims)
    with mp.get_context('spawn').Pool(n_threads) as pool:

      for curb in range(n_batches):
        cur_set = ims[(curb*n_tr)//n_batches:( (curb+1)*n_tr)//n_batches]
        split_set = split_parallel(cur_set,n_threads)
        processed_ims = pool.starmap(process_id_ims_par, [(split_set[n],conf,False,rescale) for n in range(n_threads)])
        processed_ims = merge_parallel(processed_ims)
        for ix in range(len(processed_ims)):
            zz = processed_ims[ix]
            zz = zz.astype('float32')
            zz = torch.tensor(zz).cuda()
            with torch.no_grad():
                oo = net(zz).cpu().numpy()
            preds.append(oo)

    rr = np.array(preds)
    return rr

def compute_mining_data(net, data, trk_data, rescale, confd):
  ss_t, ee_t, _ = trk_data
  ims = [dd[0] for dd in data]
  n_tr = len(data)
  a = time.time()
  t_preds = tracklet_pred(ims, net, confd, rescale)
  b = time.time()
  # print(f'Time Taken to process images {b-a}')
  # n_threads = min(24, mp.cpu_count())
  # with mp.get_context('spawn').Pool(n_threads) as pool:
  #   x_split = []
  #   for n in range(n_threads):
  #     x_split.append(list(range((n_tr*n)//n_threads,(n_tr*(n+1))//n_threads)))
  #   dists = pool.starmap(compute_dists, [ (t_preds,ss_t,ee_t,x_split[n]) for n in n_threads])
  #
  # dists = [i for sublist in dists for i in sublist]
  dists = compute_dists(t_preds,ss_t,ee_t,range(n_tr))
  c = time.time()
  # print(f'Time taken to compute dists {c-b}')
  assert [d[-1] for d in dists] == list(range(n_tr))
  overlap_dist_mean = np.array([d[3] for d in dists])
  self_dist_mean = np.array([d[2] for d in dists])
  return dists, overlap_dist_mean, self_dist_mean

def compute_dists(t_preds, ss_t, ee_t, all_xx):
  dists = []
  for xx in all_xx:
    overlap_tgts, overlap_amt = get_overlap(ss_t, ee_t, ss_t[xx], ee_t[xx], xx)
    if overlap_tgts.size > 0:
      overlap_dist = np.linalg.norm(t_preds[xx:xx + 1, :, None] - t_preds[overlap_tgts, None], axis=-1)
      overlap_mean = np.mean(overlap_dist*overlap_amt[:,None,None])
    else:
      overlap_dist = []
      overlap_mean = 2.

    self_dist = np.linalg.norm(t_preds[xx, :, None] - t_preds[xx, None], axis=-1)
    self_mean = np.mean(self_dist)
    dists.append([self_dist, overlap_dist, self_mean, overlap_mean, overlap_tgts, overlap_amt, xx])
  return dists


def train_id_classifier(all_data, conf, trks, save=False,save_file=None, bsz=16):

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
  confd.scale_factor_range = 1.1
  confd.brange = [-0.05, 0.05]
  confd.crange = [0.95, 1.05]
  rescale = conf.link_id_rescale
  n_iters = conf.link_id_training_iters
  num_times_sample = conf.link_id_mining_steps
  sampling_period = round(n_iters / num_times_sample)
  debug = conf.get('link_id_debug',False)

  logging.info('Training ID network ...')
  net.train()
  net = net.cuda()

  trk_data = []
  mining_dists = []
  for data, trk in zip(all_data,trks):
    ss, ee = trk.get_startendframes()
    tgt_id = np.array([r[1] for r in data])
    ss_t = ss[tgt_id]
    ee_t = ee[tgt_id]
    trk_data.append([ss_t,ee_t,tgt_id])
    t_dist = None
    n_tr = len(data)
    self_dist = np.ones(n_tr)
    overlap_dist = np.ones(n_tr)
    mining_dists.append([t_dist,overlap_dist, self_dist])

  train_dset = id_dset(all_data, mining_dists, trk_data, confd, rescale, valid=False, distort=True, debug=debug)
  n_workers = 10 if not debug else 0
  train_loader = torch.utils.data.DataLoader(train_dset, batch_size=bsz, pin_memory=True, num_workers=n_workers,worker_init_fn=lambda id: np.random.seed(id))
  train_iter = iter(train_loader)
  ex_ims = next(train_iter).numpy()

  hdf5storage.savemat(os.path.splitext(save_file)[0]+'_ims.mat',{'example_ims':ex_ims})

  for epoch in tqdm(range(n_iters)):

    if epoch % sampling_period == 0 and epoch > 0:
      net = net.eval()
      mining_dists = []
      for data, cur_trk_data in zip(all_data,trk_data):
        cur_dists = compute_mining_data(net, data, cur_trk_data, rescale, confd)
        mining_dists.append(cur_dists)

      net = net.train()
      del train_iter, train_loader, train_dset
      train_dset =  id_dset(all_data,mining_dists,trk_data,confd,rescale,valid=True, distort=True, debug=debug)
      train_loader = torch.utils.data.DataLoader(train_dset,batch_size=bsz,pin_memory=True,num_workers=10,worker_init_fn=lambda id: np.random.seed(id))
      train_iter = iter(train_loader)


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

  if save:
    wt_out_file = f'{save_file}-{n_iters}.p'
    torch.save({'model_state_params': net.state_dict(), 'loss_history': loss_history}, wt_out_file)

  del train_iter, train_loader, train_dset
  return net, loss_history


def link_trklet_id(linked_trks, net, mov_files, conf, all_trx, n_per_trk=50,rescale=1, min_len_select=5):

  all_data = []
  for trk, mov_file, trx in zip(linked_trks, mov_files, all_trx):
    ss, ee = trk.get_startendframes()

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
    all_data.append([data, sel_tgt, tgt_id, ss ,ee, sel_ss, sel_ee])

  net.eval()
  preds = None
  pred_map = []
  for ndx, curd in enumerate(all_data):
    data, sel_tgt, tgt_id = curd[:3]
    ims = []
    for tgt_ndx, ix in tqdm(enumerate(sel_tgt)):
      curndx = np.where(tgt_id==ix)[0][0]
      curims = data[curndx][0]
      ims.append(curims)
      pred_map.append([ndx, ix])

    cur_preds = tracklet_pred(ims, net, conf, rescale)
    assert cur_preds.shape[0] == len(sel_tgt), 'Tracklet prediction is not correct'
    if preds is None:
      preds = cur_preds
    else:
      preds = np.concatenate([preds, cur_preds],axis=0)

  pred_map = np.array(pred_map)

  logging.info('Stitching tracklets based on identity ...')

  t_info = [d[3:5] for d in all_data]
  groups = cluster_tracklets_id(preds, pred_map, t_info, conf.link_maxframes_delete)

  ids = []
  for trk, data in zip(linked_trks,all_data):
    ss, ee = data[3:5]
    cur_id = TrkFile.Tracklet(defaultval=-1, size=(1, trk.ntargets,trk.T))
    cur_id.allocate( (1,), ss-trk.T0, ee-trk.T0)
    ids.append(cur_id)

  for ndx, gr in enumerate(groups):
    for gg in gr:
      mov_ndx, trk_ndx = pred_map[gg]
      cur_id = ids[mov_ndx]
      cur_trk = linked_trks[mov_ndx]
      data = all_data[mov_ndx]
      sf,ef = data[3:5]
      cur_p = np.ones(ef[trk_ndx]-sf[trk_ndx]+1)* ndx
      cur_id.settarget(cur_p, trk_ndx, sf[trk_ndx] -cur_trk.T0, ef[trk_ndx]-cur_trk.T0)

  #   cur_tgt = min(sel_tgt[gr])
  #   for gg in gr:
  #     if sel_tgt[gg] == cur_tgt: continue
  #     match_tgt = sel_tgt[gg]
  #     trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, cur_tgt] = trk.pTrk[..., ss[match_tgt]:ee[match_tgt]+1, match_tgt]
  #     to_remove.append(match_tgt)
  #     assigned_ids[match_tgt] = cur_tgt
  #
  # # Delete the trks that have been merged
  # trk.pTrk = np.delete(trk.pTrk, to_remove, -1)
  # for k in trk.trkFields:
  #   if trk.__dict__[k] is not None:
  #     trk.__dict__[k] = np.delete(trk.__dict__[k], to_remove, -1)
  # trk.ntargets = trk.ntargets - len(to_remove)

  logging.info(f'Deleting short trajectories with length less than {conf.link_maxframes_delete}')

  params = get_default_params(conf)
  for cur_id, cur_trk in zip(ids, linked_trks):
    _, maxv = cur_id.get_min_max_val()
    nids = np.max(maxv) + 1
    t0s = np.zeros(nids, dtype=int)
    t1s = np.zeros(nids, dtype=int)
    ids_remove = []
    for id in range(nids):
      idx = cur_id.where(id)
      # idx = np.nonzero(id==ids)
      if idx[1].size>0:
        t0s[id] = np.min(idx[1])
        t1s[id] = np.max(idx[1])
      else:
        t1s[id] = -1
        ids_remove.append(id)
    isdummy = TrkFile.Tracklet(defaultval=False, size=(1, nids, cur_id.T))
    isdummy.allocate((1,), t0s, t1s)

    cur_id, ids_short = delete_short(cur_id, isdummy, params)
    _, cur_id = cur_id.unique()

    ids_left = [i for i in range(nids) if (i not in ids_short) and (i not in ids_remove)]
    cur_trk.apply_ids(cur_id)
    cur_trk.pTrkiTgt = np.array(ids_left)

  return linked_trks


def cluster_tracklets_id(embed, pred_map, t_info, min_len):

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
  n_fr = [max(sel[1]) for sel in t_info]
  tr_len = []
  for mov_ndx, trk_ndx in pred_map:
    cur_len = t_info[mov_ndx][1][trk_ndx] - t_info[mov_ndx][0][trk_ndx] + 1
    tr_len.append(cur_len)
  tr_len = np.array(tr_len)

  for ndx in range(max(F)):
    cur_gr = np.where(np.array(F) == (ndx + 1))[0]
    cur_gr_ord = np.argsort(-tr_len[cur_gr])
    cur_gr = cur_gr[cur_gr_ord]

    cur_group = []
    extra_groups = []
    ctline = [np.zeros(n) for n in n_fr]
    for cc in cur_gr:
      mov_ndx, trk_ndx = pred_map[cc]
      sel_ss = t_info[mov_ndx][0][trk_ndx]
      sel_ee = t_info[mov_ndx][1][trk_ndx]
      prev_overlap = np.sum(ctline[mov_ndx][sel_ss:sel_ee+1])/(sel_ee-sel_ss+1)
      if prev_overlap>0.05:
        if (sel_ee-sel_ss+1)>min_len:
          extra_groups.append([cc])
      else:
        cur_group.append(cc)
        ctline[mov_ndx][sel_ss:sel_ee+1] +=1

    tot_len = sum([ct.sum() for ct in ctline])
    if tot_len>min_len:
      groups.append(cur_group)
    groups.extend(extra_groups)

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
