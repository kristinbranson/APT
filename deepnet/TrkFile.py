import hdf5storage
import numpy as np
import matplotlib.pyplot as plt

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

def to_py(in_data):
  return convert(in_data,to_python=True)

def to_mat(in_data):
  return convert(in_data,to_python=False)

def load_trk(trkfile):
    trk=hdf5storage.loadmat(trkfile,appendmat=False)
    trk['issparse'] = type(trk['pTrk']) == dict
    
    # for now, convert to dense format, but don't need to do this
    if trk['issparse']:
  
      # Convert idx from matlab's fortran format to python's C format
      idx = to_py(trk['pTrk']['idx'])
      idx = np.unravel_index(idx,np.flip(trk['pTrk']['size']))
      idx = np.ravel_multi_index(idx[::-1],trk['pTrk']['size'])
      
      # default value, dtype depend on type
      if trk['pTrk']['type'] == 'nan':
        pTrk = np.zeros(trk['pTrk']['size'])
        pTrk[:] = np.nan
      elif trk['pTrk']['type'] == 'ts':
        pTrk = np.zeros(trk['pTrk']['size'])
        pTrk[:] = -np.inf
      elif trk['pTrk']['type'] == 'log':
        pTrk = np.zeros(trk['pTrk']['size'],dtype=bool)
      else:
        raise 'Unknown pTrk type %s'%trk['pTrk']['type']

      pTrk.flat[idx] = trk['pTrk']['val']
      trk['pTrk'] = pTrk
    
    # trk['pTrk'] is nlandmarks x d x nframes x maxntargets
    return trk

def save_trk(outtrkfile,trk):
  if 'issparse' in trk:
    newtrk = trk.copy()
    issparse = newtrk['issparse']
    newtrk.pop('issparse')
  else:
    issparse = False
    newtrk = trk
  
  if issparse:
    
    ps = np.array(newtrk['pTrk'].shape)
    idx_f=np.where(~np.isnan(newtrk['pTrk'].flat))[0]
    vals=newtrk['pTrk'].flat[idx_f]
  
    # Convert idx from python's C format to matlab's fortran format
    idx=np.unravel_index(idx_f,ps)
    idx=np.ravel_multi_index(idx[::-1],np.flip(ps))
    idx=to_mat(idx)
    
    newtrk['pTrk'] = {u'idx': idx,u'val': vals,u'size': ps,u'type': 'nan'}
  
  hdf5storage.savemat(outtrkfile,newtrk,appendmat=False,truncate_existing=True)

def test_sparse_load():
  trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2.trk.part'
  trk = load_trk(trkfile)

  plt.figure()
  ntargets=trk['pTrk'].shape[3]
  for t in range(5):
    for i in range(ntargets):
      plt.plot(trk['pTrk'][:,0,t,i],trk['pTrk'][:,1,t,i],'o')
  plt.show()

  outtrkfile = '200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2_test.trk'
  save_trk(outtrkfile,trk)
  
  trk2 = load_trk(outtrkfile)
  assert np.all(np.logical_or(trk2['pTrk']==trk['pTrk'],np.logical_and(np.isnan(trk2['pTrk']),np.isnan(trk['pTrk']))))
  assert np.all(np.logical_or(trk2['pTrkFrm']==trk['pTrkFrm'],np.logical_and(np.isnan(trk2['pTrkFrm']),np.isnan(trk['pTrkFrm']))))
  assert np.all(np.logical_or(trk2['pTrkTS']==trk['pTrkTS'],np.logical_and(np.isnan(trk2['pTrkTS']),np.isnan(trk['pTrkTS']))))
  assert np.all(np.logical_or(trk2['pTrkTag']==trk['pTrkTag'],np.logical_and(np.isnan(trk2['pTrkTag']),np.isnan(trk['pTrkTag']))))

if __name__=='__main__':
  # test_match_frame()
  test_sparse_load()
