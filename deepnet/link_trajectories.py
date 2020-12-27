import numpy as np
import numpy.random as random
import scipy.optimize as opt
import TrkFile
# for now I'm just using loadmat and savemat here
# when/if the format of trk files changes, then this will need to get fancier

from progressbar import progressbar

# for debugging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#plt.ion()

def match_frame(pcurr,pnext,idscurr,params,lastid=np.nan,maxcost=None):
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
    if maxcost is None:
        maxcost = params['maxcost']

    # construct the cost matrix
    # C[i,j] is the cost of matching curr[i] and next[j]
    C = np.zeros((ncurr+nnext,ncurr+nnext))
    C[:] = maxcost/2.
    C[ncurr:,nnext:] = 0
    pcurr = np.reshape(pcurr,(d*nlandmarks,ncurr,1))
    pnext = np.reshape(pnext,(d*nlandmarks,1,nnext))
    C[:ncurr,:nnext] = np.reshape(np.sum(np.abs(pcurr-pnext),axis=0),(ncurr,nnext))/nlandmarks

    # match
    idxcurr,idxnext = opt.linear_sum_assignment(C)
    costs = C[idxcurr,idxnext]
    cost = np.sum(costs)

    # idxnext < nnext, idxcurr < ncurr means we are assigning
    # an existing id
    idsnext = -np.ones(nnext,dtype=int)
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

    if params['verbose']>1:
        isdeath = np.logical_and(idxnext >= nnext,idxcurr < ncurr)
        print('N. ids assigned: %d, N. births: %d, N. deaths: %d'%(np.count_nonzero(isassigned),np.size(idxbirth),np.count_nonzero(isdeath)))
        
    return idsnext,lastid,cost,costs

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
    inter-frame match cost is minimized. Matching between frames t and t+1
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
    ids = -np.ones((maxnanimals,T),dtype=int)
    idscurr = np.arange(np.count_nonzero(idxcurr),dtype=int)
    ids[idxcurr,0] = idscurr
    lastid = np.max(idscurr)
    costs = np.zeros(T-1)

    set_default_params(params)

    for t in progressbar(range(1,T)):
        pnext = p[:,:,:,t]
        idxnext = real_idx(pnext)
        pnext = pnext[:,:,idxnext]
        idsnext,lastid,costs[t-1],_ = \
            match_frame(pcurr,pnext,idscurr,params,lastid)
        ids[idxnext,t] = idsnext
        pcurr = pnext
        idscurr = idsnext
    return ids,costs

def stitch(p,ids,params):
    """
    stitch(p,ids,params): Fill in short gaps (<= params['maxnframes_missed']) to
    connect trajectory deaths and births.
    :param p: d x nlandmarks x maxnanimals x T matrix of landmark detections
    :param ids: maxnanimals x T matrix indicating ids assigned to each detection, output of assign_ids
    :param params: parameters dict. Only relevant parameter is 'maxnframes_missed'
    :return: ids: Updated identity assignment matrix after stitching
    :return: isdummy: nids x T matrix indicating whether a frame is missed for a given id.
    """
    # ids is a maxnanimals x T
    # p is d x nlandmarks x maxnanimals x T matrix
    nids = np.max(ids)+1
    d = p.shape[0]
    nlandmarks = p.shape[1]
    T = ids.shape[1]
    
    isdummy = np.zeros((nids,T),dtype=bool)
    
    # get starts and ends for each id
    t0s = np.zeros(nids,dtype=int)
    t1s = np.zeros(nids,dtype=int)
    for id in range(nids):
        idx = np.nonzero(id==ids)
        t0s[id] = np.min(idx[1])
        t1s[id] = np.max(idx[1])
      
    allt1s = np.unique(t1s)
    assert allt1s[-1] == T-1
    # skip deaths in last frame
    for i in range(len(allt1s)-1):
        t = allt1s[i]
        # all ids that end this frame
        ids_death = np.nonzero(t1s==t)
        ids_death = ids_death[0]
        if ids_death.size == 0:
            continue
        lastid = np.max(ids_death)
        pcurr = np.zeros((d,nlandmarks,ids_death.size))
        assert np.any(isdummy[ids_death,t]) == False

        for j in range(ids_death.size):
            pcurr[:,:,j] = p[:,:,ids[:,t]==ids_death[j],t].reshape((d,nlandmarks))
        for nframes_skip in range(2,params['maxframes_missed']+2):
            # all ids that start at frame t+nframes_skip
            ids_birth = np.nonzero(t0s==t+nframes_skip)
            ids_birth = ids_birth[0]
            if ids_birth.size == 0:
                continue
            assert np.any(isdummy[ids_birth,t+nframes_skip])==False
            pnext = np.zeros((d,nlandmarks,ids_birth.size))
            for j in range(ids_birth.size):
                pnext[:,:,j]=p[:,:,ids[:,t+nframes_skip]==ids_birth[j],t+nframes_skip].reshape((d,nlandmarks))
            # try to match
            maxcost = params['maxcost_missed'][np.minimum(params['maxcost_missed'].size-1,nframes_skip-2)]
            idsnext,_,_,_= match_frame(pcurr,pnext,ids_death,params,lastid,maxcost=maxcost)
            # idsnext[j] is the id assigned to ids_birth[j]
            ismatch = idsnext <= lastid
            if np.any(ismatch)==False:
                continue
            for j in range(idsnext.size):
                id_death = idsnext[j]
                if id_death > lastid:
                    continue
                id_birth = ids_birth[j]
                ids[ids==id_birth] = id_death
                idx = np.nonzero(ids_death==id_death)
                pcurr = np.delete(pcurr,idx[0],axis=2)
                ids_death = np.delete(ids_death,idx[0])
                t0s[id_birth] = -1
                t1s[id_death] = t1s[id_birth]
                t1s[id_birth] = -1
                isdummy[id_death,t+1:t+nframes_skip] = True
                if params['verbose']>0:
                    print('Stitching id %d frame %d to id %d frame %d'%(id_death,t,id_birth,t+nframes_skip))

            if ids_death.size == 0:
                break
          
    return (ids,isdummy)

def delete_short(ids,isdummy,params):
    """
    delete_short(ids,params):
    Delete trajectories that are at most params['maxframes_delete'] frames long.
    :param ids: maxnanimals x T matrix indicating ids assigned to each detection, output of assign_ids, stitch
    :param isdummy: nids x T matrix indicating whether a frame is missed for a given id.
    :param params: parameters dict. Only relevant parameter is 'maxnframes_delete'
    :return: ids: Updated identity assignment matrix after deleting
    """
    nids=np.max(ids)+1
    T=ids.shape[1]
    
    # get starts and ends for each id
    t0s=-np.ones(nids,dtype=int)
    t1s=-np.ones(nids,dtype=int)
    nframes = np.zeros(nids)
    for id in range(nids):
        idx=np.nonzero(id==ids)
        if idx[0].size==0:
            continue
        t0s[id]=np.min(idx[1])
        t1s[id]=np.max(idx[1])
        # number of real detections
        nframes[id] = np.count_nonzero(isdummy[id,t0s[id]:t1s[id]+1]==False)
    #nframes = t1s-t0s+1
    ids_short = np.nonzero(np.logical_and(nframes <= params['maxframes_delete'],t0s>=0))
    ids_short = ids_short[0]
    ids[np.isin(ids,ids_short)] = -1
    if params['verbose'] > 0:
        print('Deleting %d short trajectories'%ids_short.size)
    return (ids,ids_short)

def delete_empty(ids):
    """
    delete_empty(ids)
    Clean up. Make ids sequential.
    :param ids: maxnanimals x T matrix indicating ids assigned to each detection, output of assign_ids, stitch, delete_short
    :return: ids: Updated identity assignment matrix after cleaning
    """
    unique_ids,newids = np.unique(ids,return_inverse=True) # should just be 0 or 1
    newids = newids - np.count_nonzero(unique_ids<0)
    newids[newids<0] = -1
    newids = np.reshape(newids,ids.shape)
    return newids

def estimate_maxcost(p,nsample=1000,prctile=95.,mult=None,nframes_skip=1):
    """
    maxcost = estimate_maxcost(p,nsample=1000,prctile=95.,mult=None)
    Estimate the threshold for the maximum cost for matching identities. This is done
    by running match_frame on some sample frames, looking at the assignment costs
    assuming all assignments are allowed, and then taking a statistic of all those
    assignment costs.
    The heuristic used is maxcost = 2.* mult .* percentile(allcosts,prctile)
    where prctile and mult are parameters
    # p is d x nlandmarks x maxnanimals x T
    """
    if mult is None:
        mult = 100./prctile
    T=p.shape[3]
    maxnanimals = p.shape[2]
    nsample = np.minimum(T,nsample)
    tsample=np.round(np.linspace(0,T-nframes_skip-1,nsample)).astype(int)
    params={}
    bignumber = np.sum(np.nanmax(p,axis=(1,2,3))-np.nanmin(p,axis=(1,2,3)))*2.1
    params['maxcost']=bignumber
    params['verbose']=0
    set_default_params(params)
    allcosts = np.zeros((maxnanimals,nsample))
    allcosts[:] = np.nan
    
    for i in range(nsample):
        t=tsample[i]
        pcurr=p[:,:,:,t]
        pnext=p[:,:,:,t+nframes_skip]
        pcurr=pcurr[:,:,real_idx(pcurr)]
        pnext=pnext[:,:,real_idx(pnext)]
        ntargets_curr = pcurr.shape[2]
        ntargets_next = pnext.shape[2]
        idscurr = np.arange(ntargets_curr)
        idsnext,_,_,costscurr=match_frame(pcurr,pnext,idscurr,params)
        ismatch = np.isin(idscurr,idsnext)
        assert np.count_nonzero(ismatch) == np.minimum(ntargets_curr,ntargets_next)
        costscurr = costscurr[:ntargets_curr]
        allcosts[:np.count_nonzero(ismatch),i] = costscurr[ismatch]
      
    isdata = np.isnan(allcosts) == False
    maxcost = mult*np.percentile(allcosts[isdata],prctile)*2.
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
    #         print('i = %d, t = %d, nmiss = %d, ncurr = %d, nnext = %d, costs removed: %s'%(i,t,nmiss,ntargets_curr,ntargets_next,str(sortedcosts[:nmiss])))

def estimate_maxcost_missed(p,maxframes_missed,nsample=1000,prctile=95.,mult=None):
    """
    maxcost_missed = estimate_maxcost_missed(p,nsample=1000,prctile=95.,mult=None)
    Estimate the threshold for the maximum cost for matching identities across > 1 frame.
    This is done by running match_frame on some sample frames, looking at the assignment costs assuming all assignments
    are allowed, and then taking a statistic of all those assignment costs.
    The heuristic used is maxcost = 2.* mult .* percentile(allcosts,prctile)
    where prctile and mult are parameters.
    # p is d x nlandmarks x maxnanimals x T
    """

    maxcost_missed = np.zeros(maxframes_missed)
    for nframes_skip in range(2,maxframes_missed+2):
        maxcost_missed[nframes_skip-2]=estimate_maxcost(p,prctile=prctile,mult=mult,nframes_skip=nframes_skip)
    return maxcost_missed

def set_default_params(params):
    if 'verbose' not in params:
        params['verbose'] = 1

def apply_ids(trk,ids):
    """
    apply_ids(trk,ids)
    Updates the trk dict according to the input id assignment. trk['pTrk'][:,:,t,i] is assigned id ids[i,t]
    :param trk: dict containing trajectories and other fields read in from a detection trk file.
    :param ids: output of assign_ids: maxnanimals x T matrix where detection [i,t] is assigned id [i,t]
    :return: newtrk: dict in the same format as trk. only the pTrk and pTrkiTgt entries are changed.
    """
    nids = np.max(ids)+1
    newtrk = trk.copy()
    newtrk['pTrkiTgt'] = np.arange(nids)+1
    # pTrk is nlandmarks x d x T x maxnanimals
    # ids is maxnanimals x T
    nlandmarks = trk['pTrk'].shape[0]
    d = trk['pTrk'].shape[1]
    T = trk['pTrk'].shape[2]
    # if isdummy is None:
    #     isdummy=np.zeros((nids,T),dtype=bool)
    # this is dense!!! should be fixed once I know what the sparse format is
    newtrk['pTrk'] = np.zeros((nlandmarks,d,T,nids))
    newtrk['pTrk'][:] = np.nan
    newtrk['pTrkTS'] = np.zeros((nlandmarks,T,nids))
    newtrk['pTrkTS'][:]=np.nan
    # tag appears to be occluded -- setting it to true for missing detections
    newtrk['pTrkTag'] = np.ones((nlandmarks,T,nids),dtype=bool)
    for id in range(nids):
        idx = np.nonzero(ids==id)
        newtrk['pTrk'][:,:,idx[1],id] = trk['pTrk'][:,:,idx[1],idx[0]]
        newtrk['pTrkTS'][:,idx[1],id] = trk['pTrkTS'][:,idx[1],idx[0]]
        newtrk['pTrkTag'][:,idx[1],id]=trk['pTrkTag'][:,idx[1],idx[0]]
    return newtrk


def test_assign_ids():
    """
    test_assign_ids():
    constructs some synthetic data and makes sure assign_ids works
    """

    # random.seed(2)
    d=2
    nlandmarks=17
    n0=6
    minn=3
    pbirth=.5
    pdeath=.5
    T=10
    maxnbirthdeath=2
    
    params={}
    params['maxcost']=.1
    # params['verbose'] = 1
    
    # create some data
    p=np.zeros((d,nlandmarks,n0,T))
    p[:]=np.nan
    ids=-np.ones((n0,T))
    
    pcurr=random.rand(d,nlandmarks,n0)
    p[:,:,:,0]=pcurr
    idscurr=np.arange(n0)
    ids[:,0]=idscurr
    lastid=np.max(idscurr)
    
    for t in range(1,T):
        
        idxcurr=real_idx(pcurr)
        ncurr=np.count_nonzero(idxcurr)
        pnext=pcurr[:,:,idxcurr]
        idsnext=idscurr
        for i in range(maxnbirthdeath):
            if ncurr>minn and random.rand(1)<=pdeath:
                pnext=pnext[:,:,:-1]
                idsnext=idsnext[:-1]
                print('%d: death'%t)
        for i in range(maxnbirthdeath):
            if random.rand(1)<=pbirth:
                lastid+=1
                pnext=np.concatenate((pnext,random.rand(d,nlandmarks,1)),axis=2)
                idsnext=np.append(idsnext,lastid)
                print('%d: birth'%t)
        nnext=pnext.shape[2]
        if nnext>p.shape[2]:
            p=np.concatenate((p,np.zeros((d,nlandmarks,nnext-p.shape[2],T))),axis=2)
            p[:,:,nnext-1,:]=np.nan
            ids=np.concatenate((ids,-np.ones((nnext-ids.shape[0],T))),axis=0)
        perm=random.permutation(nnext)
        pnext=pnext[:,:,perm]
        idsnext=idsnext[perm]
        p[:,:,:nnext,t]=pnext
        ids[:nnext,t]=idsnext
        
        pcurr=pnext
        idscurr=idsnext
    
    print('ids = ')
    print(str(ids))
    ids1,costs=assign_ids(p,params)
    
    print('assigned ids = ')
    print(str(ids1))
    print('costs = ')
    print(str(costs))


def test_match_frame():
    """
    test_match_frame():
    constructs some synthetic data and makes sure match_frame works
    """
    
    d=2
    nlandmarks=17
    ncurr=6
    nnext=ncurr+1
    
    pcurr=random.rand(d,nlandmarks,ncurr)
    pnext=np.zeros((d,nlandmarks,nnext))
    if nnext<ncurr:
        pnext=pcurr[:,:,:nnext]
    else:
        pnext[:,:,:ncurr]=pcurr
        pnext[:,:,ncurr:]=random.rand(d,nlandmarks,nnext-ncurr)
    
    idscurr=np.arange(0,ncurr)
    lastid=np.max(idscurr)
    
    perm=random.permutation(nnext)
    pnext=pnext[:,:,perm]
    
    params={}
    params['maxcost']=.8
    params['verbose']=1
    
    idsnext,lastid,cost,_=match_frame(pcurr,pnext,idscurr,params,lastid)
    print('permutation = '+str(perm))
    print('idsnext = '+str(idsnext))
    print('cost = %f'%cost)

def test_assign_ids_data():
    """
    test_assign_ids_data:
    loads data from a trkfile and runs assign_ids, stitch, delete_short, and delete_empty on them
    :return:
    """
    
    #trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2.trk.part'
    #outtrkfile='/groups/branson/bransonlab/apt/tmp/200918_m170234vocpb_m170234_odor_m170232_f0180322_full_min2_kbstitched.trk'
    
    trkfile = '/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1.trk.part'
    outtrkfile = '/groups/branson/bransonlab/apt/tmp/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1_kbstitched.trk'
    
    # parameters
    params={}
    params['verbose']=1
    params['maxframes_missed'] = 10
    params['maxframes_delete'] = 10
    params['maxcost_prctile'] = 95.
    params['maxcost_mult'] = 1.25
    params['maxcost_framesfit'] = 3
    nframes_test = np.inf

    trk = TrkFile.load_trk(trkfile)
    # frames should be consecutive
    assert np.all(np.diff(trk['pTrkFrm'],axis=1)==1), 'pTrkFrm should be consecutive frames'
    T = trk['pTrk'].shape[2]
    nlandmarks = trk['pTrk'].shape[0]
    d = trk['pTrk'].shape[1]
    # p should be d x nlandmarks x maxnanimals x T, while pTrk is nlandmarks x d x T x maxnanimals
    p = np.transpose(trk['pTrk'],(1,0,3,2))
    nframes_test = int(np.minimum(T,nframes_test))
    params['maxcost']=estimate_maxcost(p,prctile=params['maxcost_prctile'],mult=params['maxcost_mult'])
    params['maxcost_missed'] = estimate_maxcost_missed(p,params['maxcost_framesfit'],prctile=params['maxcost_prctile'],mult=params['maxcost_mult'])
    print('maxcost set to %f'%params['maxcost'])
    ids,costs = assign_ids(p[:,:,:,:nframes_test],params)
    nids_original = np.max(ids)+1
    ids,isdummy = stitch(p[:,:,:,:nframes_test],ids,params)
    ids,ids_short = delete_short(ids,isdummy,params)
    ids = delete_empty(ids)
    
    newtrk = apply_ids(trk,ids)
    
    # save to file
    TrkFile.save_trk(outtrkfile,newtrk)
    
    plt.figure()
    plt.subplot(211)
    for i in range(ids.shape[0]):
        plt.plot(ids[i,:])
    plt.subplot(212)
    plt.plot(costs)
    
    plt.figure()
    nids = newtrk['pTrk'].shape[3]
    print('%d ids in %d frames, removed %d ids'%(nids,nframes_test,nids_original-nids))
    nidsplot = np.minimum(nids,10)
    minp = np.nanmin(newtrk['pTrk'][newtrk['pTrk']!=-np.inf])
    maxp = np.nanmax(newtrk['pTrk'][newtrk['pTrk']!=np.inf])
    for id in range(nidsplot):
        ts = np.nonzero(np.isnan(newtrk['pTrk'][0,0,:,id])==False)
        for i in range(d):
            hax = plt.subplot(nidsplot,d,i+id*d+1)
            for j in range(nlandmarks):
                h = plt.plot(newtrk['pTrk'][j,i,:,id])
                color = h[0].get_color()
                plt.plot(ts[0][[0,-1]],newtrk['pTrk'][j,i,ts[0][[0,-1]],id],'o',color=color,mfc=color)
            plt.title('id = %d, coord %d, t0 = %d, t1 = %d'%(id,i,ts[0][0],ts[0][-1]))
            hax.set_xlim((0,nframes_test))
            hax.set_ylim((minp,maxp))
    plt.show()
    
def test_estimate_maxcost():
    
    trkfile='/groups/branson/home/kabram/temp/roian_multi/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1.trk.part'
    outtrkfile='/groups/branson/bransonlab/apt/tmp/200918_m170234vocpb_m170234_odor_m170232_f0180322_full1_kbstitched.trk'
    
    # parameters
    maxcost_prctile=95.
    params={}
    params['verbose']=1
    params['maxframes_missed']=10
    params['maxframes_delete']=10
    
    trk=TrkFile.load_trk(trkfile)
    # frames should be consecutive
    assert np.all(np.diff(trk['pTrkFrm'],axis=1)==1),'pTrkFrm should be consecutive frames'
    T=trk['pTrk'].shape[2]
    nlandmarks=trk['pTrk'].shape[0]
    d=trk['pTrk'].shape[1]
    # p should be d x nlandmarks x maxnanimals x T, while pTrk is nlandmarks x d x T x maxnanimals
    p=np.transpose(trk['pTrk'],(1,0,3,2))
    
    maxcost = np.zeros(params['maxframes_missed']+1)
    maxcost[0] = estimate_maxcost(p,prctile=maxcost_prctile)
    maxcost[1:] = estimate_maxcost_missed(p,params,prctile=maxcost_prctile)
    plt.figure()
    plt.plot(np.arange(params['maxframes_missed']+1)+1,maxcost,'o-')
    plt.show()
    
if __name__ == '__main__':
    #test_match_frame()
    test_assign_ids_data()
    #test_estimate_maxcost()
    
