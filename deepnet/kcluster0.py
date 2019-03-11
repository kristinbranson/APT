import numpy as num
import scipy.cluster.vq as vq
#import scipy.linalg.decomp as decomp
#from version import DEBUG, DEBUG_REPEATABLE_BEHAVIOR

DEBUG = False
DEBUG_REPEATABLE_BEHAVIOR = False
d = 2

def clusterdistfun(x,c):
    n = x.shape[0]
    nclusts = c.shape[0]
    D = num.empty((nclusts,n))
    for i in range(nclusts):
        D[i,:] = num.sum((x - c[i,:])**2,axis=1)
    return D

def furthestfirst(x,k,mu0=None,start='mean'):
    # number of data points
    n = x.shape[0]
    # data dimensionality: always 2
    # d = x.shape[1]
    
    # returned centers
    mu = num.empty((k,d))
    
    # initialize first center
    if num.any(mu0) == False:
        if start == 'mean':
            # choose the first mean to be the mean of all the data
            mu[0,:] = num.mean(x,axis=0)
        else:
            # choose a random point to be the first center
            mu[0,:] = x[num.random.randint(0,x.shape[0],1)[0],:]
    else:
        mu[0,:] = mu0
    
    # initialize distance to all centers to be infinity
    Dall = num.empty((k,n))
    Dall[:] = num.inf

    for i in range(1,k):
        # compute distance to all centers from all points
        Dall[i-1,:] = num.sum((x - mu[i-1,:])**2,axis=1)
        # compute the minimum distance from all points to the centers
        D = num.amin(Dall,axis=0)
        # choose the point furthest from all centers as the next center
        j = num.argmax(D)
        mu[i,:] = x[j,:]

    Dall[k-1,:] = num.sum((x - mu[k-1,:])**2,axis=1)
    idx = num.argmin(Dall,axis=0)

    return (mu,idx)

def gmminit(x,k,weights=None,kmeansiter=20,kmeansthresh=.001):

    n = x.shape[0]
    # 2d
    #d = x.shape[1]

    # initialize using furthest-first clustering
    (mu,idx) = furthestfirst(x,k,start='random')

    # use k-means, beginning from furthest-first
    (mu,dmin) = vq.kmeans(x,k,iter=kmeansiter,thresh=kmeansthresh)

    # get labels for each data point
    D = clusterdistfun(x,mu)
    idx = num.argmin(D,axis=0)

    # allocate covariance and priors
    S = num.empty((d,d,k))
    priors = num.empty(k)

    if num.any(weights) == False:
        # unweighted
        for i in range(k):
            # compute prior for each cluster
            nidx = num.sum(num.double(idx==i))
            priors[i] = nidx
            # compute mean for each cluster
            mu[i,:] = num.mean(x[idx==i,:],axis=0)
            # compute covariance for each cluster
            diffs = x[idx==i,:] - mu[i,:].reshape(1,d)
            S[:,:,i] = num.dot(num.transpose(diffs),diffs) / priors[i]
    else:
        # replicate weights
        weights = weights.reshape(n,1)
        for i in range(k):
            # compute prior for each cluster
            nidx = num.sum(num.double(idx==i))
            priors[i] = num.sum(weights[idx==i])
            # compute mean for each cluster
            mu[i,:] = num.sum(weights[idx==i]*x[idx==i,:],axis=0)/priors[i]
            # compute covariance for each cluster
            diffs = x[idx==i,:] - mu[i,:].reshape(1,d)
            diffs *= num.sqrt(weights[idx==i])
            S[:,:,i] = num.dot(num.transpose(diffs),diffs) / priors[i]

    # normalize priors
    priors = priors / num.sum(priors)

    # return
    return (mu,S,priors)

def gmm(x,k,weights=None,nreplicates=10,kmeansiter=20,kmeansthresh=.001,emiters=100,emthresh=.001,mincov=.01):

    # for debugging only: reseed the random number generator at 0 for repeatable behavior
    if DEBUG_REPEATABLE_BEHAVIOR:
        num.random.seed(0)

    # number of data points
    n = x.shape[0]
    # dimensionality of each data point: 2d
    #d = x.shape[1]
    # initialize min error
    minerr = num.inf

    # replicate many times
    for rep in range(nreplicates):

        # initialize randomly
        (mu,S,priors) = gmminit(x,k,weights,kmeansiter,kmeansthresh)

        #print "mu is "
        #print(mu)

        # optimize fit using EM
        (mu,S,priors,gamma,err) = gmmem(x,mu,S,priors,weights,emiters,emthresh,mincov)

        if rep == 0 or err < minerr:
            mubest = mu
            Sbest = S
            priorsbest = priors
            minerr = err
            gammabest = gamma

    return (mubest,Sbest,priorsbest,gammabest,minerr)

def gmmmemberships(mu,S,priors,x,weights=1,initcovars=None):
    if initcovars is None:
        initcovars = S.copy()

    # number of data points
    n = x.shape[0]
    # dimensionality of data: 2d
    #d = x.shape[1]
    # number of clusters
    k = mu.shape[0]

    #print "k is {}".format(k)

    # allocate output
    gamma = num.empty((n,k))

    # normalization constant
    normal = 2*num.pi
    #normal = (2.0*num.pi)**(num.double(d)/2.0)
    #print 'd = %d, normal = %f' % (d,normal)
    #print 'mu = '
    #print mu
    #print 'S = '
    #for j in range(k):
    #    print S[:,:,j]
    #print 'priors = '
    #print priors
    #print 'weights = '
    #print weights

    # compute the activation for each data point
    for j in range(k):
        #print 'j = %d' % j
        #print 'mu[%d,:] = '%j + str(mu[j,:])
        #print 'S[:,:,%d] = '%j + str(S[:,:,j])
        diffs = x - mu[j,:]
        #print 'diffs = '
        #print diffs

        zz = S[0,0,j]*S[1,1,j] - S[0,1,j]**2
        if zz <= 0:
            if DEBUG: print 'S[:,:,%d] = '%j + str(S[:,:,j]) + ' is singular'
            if DEBUG: print 'Reverting to initcovars[:,:,%d] = '%j + str(initcovars[:,:,j])
            S[:,:,j] = initcovars[:,:,j]
            zz = S[0,0,j]*S[1,1,j] - S[0,1,j]**2

        temp = (diffs[:,0]**2*S[1,1,j]
                - 2*diffs[:,0]*diffs[:,1]*S[0,1,j]
                + diffs[:,1]**2*S[0,0,j]) / zz

        gamma[:,j] = num.exp(-.5*temp)/(normal*num.sqrt(zz))
        #gamma[:,j] = num.exp(-.5*temp)/(normal*num.prod(num.diag(c)))
        #print 'gamma[:,%d] = ' % j
        #print gamma[:,j]

    # include prior
    gamma *= priors
    #print 'after including prior, gamma = '
    #print gamma

    # compute negative log likelihood
    e = -num.sum(num.log(num.sum(gamma,axis=1))*weights)
    
    s = num.sum(gamma,axis=1)
    #print 's = '
    #print s
    # make sure we don't divide by 0
    s[s==0] = 1
    gamma /= s.reshape(n,1)
    #print 'gamma = '
    #print gamma
    
    return (gamma,e)    

def gmmupdate(mu,S,priors,gamma,x,weights=1,mincov=.01,initcovars=None):

    if num.any(initcovars) == False:
        initcovars = S
    
    # number of data points
    n = gamma.shape[0]
    # number of clusters
    k = gamma.shape[1]
    # dimensionality of data: 2d
    #d = x.shape[1]

    gamma *= weights.reshape(n,1)

    # update the priors (note that it has not been normalized yet)
    priors = num.sum(gamma,axis=0)
    Z = priors.copy()
    sumpriors = num.sum(priors)
    if sumpriors > 0:
        priors /= sumpriors
        issmall = priors < .01
        issmall = issmall.any()
    else:
        if DEBUG: print "All priors are too small, reinitializing"
        issmall = True # num.bool_ NOT bool
    #print 'updated priors to ' + str(priors)

    # if any prior is to small, then reinitialize that cluster
    if issmall:
        fixsmallpriors(x,mu,S,priors,initcovars,gamma)
        priors = num.sum(gamma,axis=0)
        Z = priors.copy()
        priors /= num.sum(priors)
        if DEBUG: print 'after fixsmallpriors, priors is ' + str(priors)

    #issmall = issmall.any()
    if issmall:
        if DEBUG: 
            print 'outside fixsmallpriors'
            print 'reset mu = ' + str(mu)
            for i in range(k):
                print 'reset S[:,:,%d] = '%i + str(S[:,:,i])
            print 'reset priors = ' + str(priors)
        #print 'reset gamma = '
        #print gamma

    for i in range(k):
        # update the means
        mu[i,:] = num.sum(gamma[:,i].reshape(n,1)*x,axis=0)/Z[i]
        if DEBUG:
            if issmall: 
                print 'updated mu[%d,:] to '%i + str(mu[i,:])
        # update the covariances
        diffs = x - mu[i,:]
        #if issmall: print 'diffs = ' + str(diffs)
        diffs *= num.sqrt(gamma[:,i].reshape(n,1))
        #if issmall: print 'weighted diffs = ' + str(diffs)
        S[:,:,i] = (num.dot(num.transpose(diffs),diffs)) / Z[i]
        if DEBUG:
            if issmall: 
                print 'updated S[:,:,%d] to [%.4f,%.4f;%.4f,%.4f]'%(i,S[0,0,i],S[0,1,i],S[1,0,i],S[1,1,i])
        # make sure covariance is not too small
        if mincov > 0:
            # hard-coded 2x2
            eigval_T = S[0,0,i] + S[1,1,i]
            mineigval = eigval_T - num.sqrt(eigval_T**2/4 - S[1,1,i])
            if mineigval < mincov:
                S[:,:,i] = initcovars[:,:,i]
                if DEBUG: 
                    print 'mineigval = %.4f'%mineigval
                    print 'reinitializing covariance'
                    print 'initcovars[:,:,%d] = [%.4f,%.4f;%.4f,%.4f]'%(i,initcovars[0,0,i],initcovars[0,1,i],initcovars[1,0,i],initcovars[1,1,i])
                
def gmmem(x,mu0,S0,priors0,weights=None,niters=100,thresh=.001,mincov=.01):

    #print 'mu = '
    #print mu0
    mu = mu0.copy()
    S = S0.copy()
    priors = priors0.copy()

    e = num.inf
    # store initial covariance in case covariance becomes too small
    if mincov > 0:
        for i in range(S.shape[2]):
            #print 'S initially is: '
            #print S[:,:,i]
            eigval_T = S[0,0,i] + S[1,1,i]
            mineigval = eigval_T - num.sqrt(eigval_T**2/4 - S[1,1,i])
            if num.isnan(mineigval) or mineigval < mincov:
                D, U = num.linalg.eig(S[:,:,i])
                if DEBUG: print "initial S[:,:,%d] is singular"%i
                if DEBUG: print "S[:,:,%d] = "%i
                if DEBUG: print str(S[:,:,i])
                D[D<mincov] = mincov
                S[:,:,i] = num.dot(num.dot(U,num.diag(D)),U.transpose())
                if DEBUG: print 'S[:,:,%d] reinitialized to: '%i
                if DEBUG: print S[:,:,i]
    initcovars = S.copy()

    for iter in range(niters):

        # E-step: compute memberships
        [gamma,newe] = gmmmemberships(mu,S,priors,x,weights,initcovars)

        # M-step: update parameters
        gmmupdate(mu,S,priors,gamma,x,weights,mincov,initcovars)

        # if we've converged, break
        if newe >= e - thresh:
            break
        
        e = newe

    [gamma,e] = gmmmemberships(mu,S,priors,x,weights,initcovars)

    return (mu,S,priors,gamma,e)

def fixsmallpriors(x,mu,S,priors,initcovars,gamma):

    #print 'calling fixsmallpriors with: '
    #print 'mu = ' + str(mu)
    #print 'S = '
    #for i in range(S.shape[2]):
    #    print S[:,:,i]
    #print 'priors = ' + str(priors)
    #for i in range(initcovars.shape[2]):
    #    print 'initcovars[:,:,%d]: '%i
    #    print initcovars[:,:,i]
    #    print 'initcovars[:,:,%d].shape: '%i + str(initcovars[:,:,i].shape)
        
    MINPRIOR = .01
    issmall = priors < .01
    if not issmall.any():
        return

    n = x.shape[0]
    # 2d
    #d = x.shape[1]
    k = mu.shape[0]

    # loop through all small priors
    smalli, = num.where(issmall)
    for i in smalli:

        if DEBUG: print 'fixing cluster %d with small prior = %f: '%(i,priors[i])

        # compute mixture density of each data point
        p = num.sum(gamma*priors,axis=1)

        #print 'samples: '
        #print x
        #print 'density of each sample: '
        #print p

        # choose the point with the smallest probability
        j = p.argmin()

        if DEBUG: print 'lowest density sample: x[%d] = '%j + str(x[j,:])

        # create a new cluster
        mu[i,:] = x[j,:]
        S[:,:,i] = initcovars[:,:,i]
        priors *= (1. - MINPRIOR)/(1.-priors[i])
        priors[i] = MINPRIOR

        if DEBUG: 
            print 'reset cluster %d to: '%i
            print 'mu = ' + str(mu[i,:])
            print 'S = '
            print S[:,:,i]
            print 'S.shape: ' + str(S[:,:,i].shape)
            print 'priors = ' + str(priors)

        # update gamma
        [gamma,newe] = gmmmemberships(mu,S,priors,x,1,initcovars)

#n0 = 100
#n1 = 100
#d = 3
#k = 2
#mu0 = num.array([0.,0.,0.])
#mu1 = num.array([5.,5.,5.])
#weights = num.hstack((10*num.ones(n0),num.ones(n1)))
#std0 = 1
#std1 = 1
#x0 = num.random.standard_normal(size=(n0,d))*std0 + mu0
#x1 = num.random.standard_normal(size=(n1,d))*std1 + mu1
#x = num.vstack((x0,x1))
#(mu,S,priors,err) = gmm(x,k,weights=weights)
#print 'centers = '
#print mu
#print 'covars = '
#print S[:,:,0]
#print S[:,:,1]
#print 'priors = '
#print priors
#print 'err = %f' % err
