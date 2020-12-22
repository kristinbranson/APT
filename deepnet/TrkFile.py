import hdf5storage

def load_trk(trkfile):
    trk=hdf5storage.loadmat(trkfile,appendmat=False)
    trk['issparse'] = type(trk['pTrk']) == dict
    if trk['issparse']:
      trk['pTrk'] = SparseLabelArray.full(trk['pTrk'])
    # trk['pTrk'] is nlandmarks x d x nframes x maxntargets
    return trk

def save_trk(outtrkfile,trk):
  hdf5storage.savemat(outtrkfile,trk,appendmat=False,truncate_existing=True)