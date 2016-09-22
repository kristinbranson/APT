classdef CPRPrune
  
  properties
    xy4d % [nRep x 2 x nFrm x npts]
    imnr % number rows image
    imnc % number cols image
    
    sigD % sigma for distance function
  end
  
  methods
    
    function obj = CPRPrune(xy4d,sigD)
%       [nfrm,nrep,nfidsTimes2] = size(trkpfull);
%       nfids = nfidsTimes2/2;
%       xy = reshape(trkpfull,nfrm,nrep,nfids,2);
%       obj.xy4d = permute(xy,[2 4 1 3]);
      obj.xy4d = xy4d;
%       obj.imnr = 256;
%       obj.imnc = 256;
      obj.sigD = sigD;
    end
    
    function [pTrk,pTrkAbs,pbest,pprevloc,p0sigs] = run(obj,Tend,iPt)
      % pTrk: [Tx2]. (x,y) for optimal track using back-propagation.
      % pTrkAbs: [Tx2]. (x,y) for track simply following maxima of pbest.
      %
      % pbest: [nr,nc,T]. pbest(i,j,t) gives the maximum/best probability of reaching (xgrid(i,j),ygrid(i,j)) at time t. [sum of pbest(:,:,t)] ~ 1.
      % pprev: [nr,nc,T]. pprev(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to pbest(i,j,t).
      % p0sigs: [T,2]; % 2D bandwidth

      nr = obj.imnr;
      nc = obj.imnc;
      T = Tend;
      %T = size(obj.xy4d,3); % note: T here is frame num, NOT CPR iteration as during regression
      
      [xgrid,ygrid] = meshgrid(1:nc,1:nr);
      xy3d = obj.xy4d(:,:,1:T,iPt); % [nRep x 2 x T]
      
      % precompute Dmat
      [xgridDbigmat,ygridDbigmat] = meshgrid(-nc:1:nc,-nr:1:nr);
      Dbigmat = CPRPrune.computeD(xgridDbigmat,ygridDbigmat,0,0,obj.sigD);
      assert(isequal(size(Dbigmat),[2*nr+1,2*nc+1]));
      
      % initialize 
      pbest = nan(nr,nc,T); % pbest(i,j,t) gives the maximum/best probability of reaching (xgrid(i,j),ygrid(i,j)) at time t. [sum of pbest(:,:,t)] ~ 1.
      pprevloc = nan(nr,nc,T); % pprevloc(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to pbest(i,j,t).
      p0sigs = nan(T,2); % 2D bandwidth
      
      [p0_1,p0sigs(1,:)] = obj.estimateP0(xy3d(:,:,1),xgrid,ygrid);
      pbest(:,:,1) = p0_1;
      pprevloc(:,:,1) = nan; 
      
      % iter
      for t=2:T
        
        [p0_t,p0sigs(t,:)] = obj.estimateP0(xy3d(:,:,t),xgrid,ygrid);
        Amat = obj.computeAmat(p0_t,xgrid,ygrid,obj.sigD); % Amat indexed by r_(t-1)
 
        % loop over x_t
        for i=1:nr
        for j=1:nc
          x_t = xgrid(i,j);
          y_t = ygrid(i,j);
          p_tm1 = pbest(:,:,t-1);
          
          % for this pt (x_t,y_t): maximize P_t((x_t,y_t),{observed reps at t} |
          % (x_[t-1],y_[t-1])) * P_[t-1](x_[t-1],y_[t-1]) over (x_[t-1],y_[t-1])
          
          % CHECK THIS WHEN SIGD IS NOT INF
          % Distance weight of (x_[t-1],y_[t-1]) relative to (x_t,y_t)
          % if (x_t,y_t)==(1,1), then Dmat=Dbigmat(nr+1:end-1,nc+1:end-1)
          % if (x_t,y_t)==(2,2), then Dmat=Dbigmat(nr:end-2,nc:end-2)
          % if (x_t,y_t)==(nc,1), then Dmat=Dbigmat(nr+1:end-1,2:nc+1)
          % if (x_t,y_t)==(1,nr), then Dmat=Dbigmat(2:nr+1,nc+1:end-1)
          rowStart = nr+2-y_t;
          colStart = nc+2-x_t;
          Dmat = Dbigmat(rowStart:rowStart+nr-1,colStart:colStart+nc-1);
          assert(isequal(size(Dmat),[nr nc]));
          assert(Dmat(i,j)==1);
          % Dmat is conceptually indexed by x_[t-1],y_[t-1]
          %Dmat = CPRPrune.computeD(xgrid,ygrid,x_t,y_t,obj.sigD);
          p_t_givenRepsAnd_p_tm1 = Amat.*p0_t(i,j).*Dmat.*p_tm1; % indexed by x_[t-1],y_[t-1]
          [pbest(i,j,t),pprevloc(i,j,t)] = max(p_t_givenRepsAnd_p_tm1(:));
        end
        end
        
        % pbest(:,:,t) are now all assigned. However the values are not
        % normalized. I think this is right, there is no constraint that
        % sum(pbest)==1.
        tmp = pbest(:,:,t);
        assert(all(tmp(:)>=0));
        tmpsum = sum(tmp(:));
        pbest(:,:,t) = pbest(:,:,t)/tmpsum;
        
        fprintf('Done with t=%d\n',t);
      end
      
      % Find tracks
      pTrk = nan(T,2);
      pTrkAbs = nan(T,2);
      for t = T:-1:1
        pbestCurr = pbest(:,:,t);

        % pTrkAbs
        [~,idx] = max(pbestCurr(:));
        [rbest,cbest] = ind2sub(size(pbestCurr),idx);
        pTrkAbs(t,:) = [cbest rbest];
        
        % pTrk
        if t==T
          pTrk(T,:) = [cbest rbest];
        end
        if t>1
          ct = pTrk(t,1);
          rt = pTrk(t,2);          
          idxPrev = pprevloc(rt,ct,t);
          [rprev,cprev] = ind2sub(size(pbestCurr),idxPrev);
          pTrk(t-1,:) = [cprev rprev];
        end
      end
    end
    
  end
  
  methods (Static)
    
    function [p0,u] = estimateP0(xy,xgrid,ygrid)
      % Estimate default PDF using KDE
      %
      % xy: [Nx2] points.
      % xgrid,ygrid: [nr x nc].
      %
      % p0: [nr x nc] estimated PDF for xy. sum(p0(:))~1 if the xy
      % distribution falls "within" xgrid,ygrid.
      % u: KDE bandwidth used in estimate
      
      [p0,~,u] = ksdensity(xy,[xgrid(:) ygrid(:)]);
      p0 = reshape(p0,size(xgrid));
    end
    
    function Amat = computeAmat(p0,xgrid,ygrid,sigD)
      % Compute A assuming (x0,y0) at all grid points
      
      [nr,nc] = size(p0);
      Amat = nan(size(p0));
      for i=1:nr
      for j=1:nc
        Amat(i,j) = CPRPrune.computeA(p0,xgrid,ygrid,xgrid(i,j),ygrid(i,j),sigD);
      end
      end
    end
    
    function A = computeA(p0,xgrid,ygrid,x0,y0,sigD)
      % Compute A, normalization constant for conditional prob
      %
      % p0: [nr x nc] P0 pdf
      % xgrid/ygrid: [nr x nc] x-coord array meshgrid style for p0
      %
      % x0/y0: (x,y) coords of "previous" loc/coord
      % sigD: scalar, SD for distance function
      
      D = CPRPrune.computeD(xgrid,ygrid,x0,y0,sigD);      
      prod = p0.*D;
      A = 1/sum(prod(:));
      % [sum_over_(x,y)_for_given_(x0,y0) A*p*D]==1
    end
    
    function D = computeD(xgrid,ygrid,xstar,ystar,sig)
      % Compute exponential distance weight from points in xgrid, ygrid to
      % single point (xstar,ystar)
      %
      % xgrid/ygrid: [nrxnc]
      % xstar, ystar, scalars (typically falling 'within' range of
      % xgrid,ygrid
      % sig: SD to use for distance weight
      %
      % D: [nrxnc]
      
      d2 = (xgrid-xstar).^2 + (ygrid-ystar).^2;
      D = exp(-d2/2/sig^2);
    end
    
  end
      
end