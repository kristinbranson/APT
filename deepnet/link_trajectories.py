import numpy as np
import numpy.random as random
import scipy.optimize as opt
import pdb

def match_frame(pcurr,pnext,idscurr,params,lastid=np.nan):
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
    d = pcurr.shape[0]
    nlandmarks = pcurr.shape[1]
    ncurr = pcurr.shape[2]
    nnext = pnext.shape[2]
    assert pnext.shape[0] == d, \
        'Dimensions do not match, curr = %d, next = %d'%(d,pnext.shape[0])
    assert pnext.shape[1] == nlandmarks, \
        'N landmarks do not match, curr = %d, next = %d'%(nlandmarks,pnext.shape[1])

    # construct the cost matrix
    # C[i,j] is the cost of matching curr[i] and next[j]
    C = np.zeros((ncurr+nnext,ncurr+nnext))
    C[:] = params['maxcost']/2.
    C[ncurr:,nnext:] = 0
    pcurr = np.reshape(pcurr,(d*nlandmarks,ncurr,1))
    pnext = np.reshape(pnext,(d*nlandmarks,1,nnext))
    C[:ncurr,:nnext] = np.reshape(np.sum(np.abs(pcurr-pnext),axis=0),(ncurr,nnext))/nlandmarks

    # match
    idxcurr,idxnext = opt.linear_sum_assignment(C)
    cost = np.sum(C[idxcurr,idxnext])

    # idxnext < nnext, idxcurr < ncurr means we are assigning
    # an existing id
    idsnext = -np.ones(nnext)
    isassigned = np.logical_and(idxnext < nnext,idxcurr < ncurr)    
    idsnext[idxnext[isassigned]] = idscurr[idxcurr[isassigned]]

    # idxnext < nnext, idxcurr >= ncurr means we are creating
    # a new trajectory
    if np.isnan(lastid):        
        lastid = np.max(idscurr)
    idxbirth = idxnext[np.logical_and(idxnext < nnext,idxcurr >= ncurr)]
    for i in range(np.size(idxbirth)):
        lastid += 1
        idsnext[idxbirth[i]] = lastid

    if params['verbose']>0:
        isdeath = np.logical_and(idxnext >= nnext,idxcurr < ncurr)
        print('N. ids assigned: %d, N. births: %d, N. deaths: %d'%(np.count_nonzero(isassigned),np.size(idxbirth),np.count_nonzero(isdeath)))
        
    return idsnext,lastid,cost

def real_idx(pcurr):
    """
    real_idx(pcurr)
    Helper function that determines which indices of pcurr correspond
    to real detections. Convention is that ALL coordinates will be nan 
    if this is a dummy detection.
    Input: pcurr is d x nlandmarks x nanimals
    Returns binary array idx of length nanimals which is True iff the 
    coordinates for that animal are not the dummy value.
    """

    idx = np.all(np.isnan(pcurr),axis=(0,1))==False
    return idx

def assign_ids(p,params):
    """
    assign_ids(p,params)
    Assign identities to each detection in each frame so that one-to-one 
    inter-fram match cost is minimized. Matching between frames t and t+1
    is done using match_frame. 
    Input:
    p: d x nlandmarks x maxnanimals x T matrix, where p[:,:,:,t] are the
    detections for frame t. All coordinates will be nan if the number of 
    detections in a given frame is less than maxnanimals. 
    params: dictionary of parameters (see match_frame for details). 
    Output: ids is a maxnanimals x T matrix with integers 0, 1, ... 
    indicating the identity of each detection in each frame. -1 is assigned
    to dummy detections. 
    """
    
    # p is d x nlandmarks x maxnanimals x T
    # nan is used to indicate missing data
    d = p.shape[0]
    nlandmarks = p.shape[1]
    maxnanimals = p.shape[2]
    T = p.shape[3]
    pcurr = p[:,:,:,0]
    idxcurr = real_idx(pcurr)
    pcurr = pcurr[:,:,idxcurr]
    ids = -np.ones((maxnanimals,T))
    idscurr = np.arange(np.count_nonzero(idxcurr))
    ids[idxcurr,0] = idscurr
    lastid = np.max(idscurr)
    costs = np.zeros(T-1)

    set_default_params(params)
    
    for t in range(1,T):
        pnext = p[:,:,:,t]
        idxnext = real_idx(pnext)
        pnext = pnext[:,:,idxnext]
        idsnext,lastid,costs[t-1] = \
            match_frame(pcurr,pnext,idscurr,params,lastid)
        ids[idxnext,t] = idsnext
        pcurr = pnext
        idscurr = idsnext
    return ids,costs

def set_default_params(params):
    if 'verbose' not in params:
        params['verbose'] = 1

def test_assign_ids():
    """
    test_assign_ids():
    constructs some synthetic data and makes sure assign_ids works
    """

    #random.seed(2)
    d = 2
    nlandmarks = 17
    n0 = 6
    minn = 3
    pbirth = .5
    pdeath = .5
    T = 10
    maxnbirthdeath = 2

    params = {}
    params['maxcost'] = .1
    #params['verbose'] = 1

    # create some data
    p = np.zeros((d,nlandmarks,n0,T))
    p[:] = np.nan
    ids = -np.ones((n0,T))
    
    pcurr = random.rand(d,nlandmarks,n0)
    p[:,:,:,0] = pcurr
    idscurr = np.arange(n0)
    ids[:,0] = idscurr
    lastid = np.max(idscurr)
    
    for t in range(1,T):

        idxcurr = real_idx(pcurr)
        ncurr = np.count_nonzero(idxcurr)
        pnext = pcurr[:,:,idxcurr]
        idsnext = idscurr
        for i in range(maxnbirthdeath):
            if ncurr > minn and random.rand(1) <= pdeath:
                pnext = pnext[:,:,:-1]
                idsnext = idsnext[:-1]
                print('%d: death'%t)
        for i in range(maxnbirthdeath):
            if random.rand(1) <= pbirth:
                lastid += 1
                pnext = np.concatenate((pnext,random.rand(d,nlandmarks,1)),axis=2)
                idsnext = np.append(idsnext,lastid)
                print('%d: birth'%t)
        nnext = pnext.shape[2]
        if nnext > p.shape[2]:
            p = np.concatenate((p,np.zeros((d,nlandmarks,nnext-p.shape[2],T))),axis=2)
            p[:,:,nnext-1,:] = np.nan
            ids = np.concatenate((ids,-np.ones((nnext-ids.shape[0],T))),axis=0)
        perm = random.permutation(nnext)
        pnext = pnext[:,:,perm]
        idsnext = idsnext[perm]
        p[:,:,:nnext,t] = pnext
        ids[:nnext,t] = idsnext

        pcurr = pnext
        idscurr = idsnext

    print('ids = ')
    print(str(ids))
    ids1,costs = assign_ids(p,params)

    print('assigned ids = ')
    print(str(ids1))
    print('costs = ')
    print(str(costs))
    
        
def test_match_frame():
    """
    test_match_frame():
    constructs some synthetic data and makes sure match_frame works
    """

    d = 2
    nlandmarks = 17
    ncurr = 6
    nnext = ncurr+1

    pcurr = random.rand(d,nlandmarks,ncurr)
    pnext = np.zeros((d,nlandmarks,nnext))
    if nnext < ncurr:
        pnext = pcurr[:,:,:nnext]
    else:
        pnext[:,:,:ncurr] = pcurr
        pnext[:,:,ncurr:] = random.rand(d,nlandmarks,nnext-ncurr)

    idscurr = np.arange(0,ncurr)
    lastid = np.max(idscurr)

    perm = random.permutation(nnext)
    pnext = pnext[:,:,perm]

    params = {}
    params['maxcost'] = .8
    params['verbose'] = 1

    idsnext,lastid,cost = match_frame(pcurr,pnext,idscurr,params,lastid)
    print('permutation = '+str(perm))
    print('idsnext = '+str(idsnext))
    print('cost = %f'%cost)
    
if __name__ == '__main__':
    #test_match_frame()
    test_assign_ids()
