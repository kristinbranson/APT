classdef CPRPrune < handle
  
  % AL20161206. Well something seems wrong with this
  % implementation/formulation but I am not sure what. One clear issue is
  % that summing over all possible p_tminus1 positions to generate the pdf 
  % for p_t seems better than just taking the max/best p_tm1. Eg imagine a
  % bimodal p_tm1 distribution with one narrow sharp peak and one much
  % larger but shallower peak. (A) Just taking the max does not adequately
  % represent the total probability for arriving at p_t; (B) taking the 
  % argmax does not accurately represent the "previous pointer" or
  % where-did-I-most-likely-come-from. (A) can be addressed by summing over 
  % all possible p_tm1 locations. Don't know what to do about (B).
  %
  % The seemingly deeper issue is that I am encountering a situation where 
  % I have fixed a certain location x and am looking at the distribution 
  % over previous locations, ie Q=p_t(x|y)*p_tm1(y). The location x is 
  % located between two peaks in p_tm1, one small and one large, yet Q is 
  % largely centered on the smaller peak. Numerically this seems to arise 
  % due to the normalization constant A, which is much larger near the 
  % small peak. However physically the situation makes no sense -- summing 
  % the distribution near the small peak shows that it is overwhelmingly more
  % likely that y(at t-1) is located near the small peak, vs anywhere else
  % (eg near the large peak). That is the contribution to Q seems to come
  % wholly from the small peak. This seems unphysical and I suspect the
  % mathematical formulation must be wrong somehow. I may also be getting
  % confused about probabilities vs likelihoods etc.
  
  properties
    imnr % number rows image
    imnc % number cols image
    sigD % sigma for distance function
    bigD % precomputed big D matrix 

    xy % [nRep x 2 x nFrm] shifted xy coords for pt of interest, all replicates/frames    
    roi % [1x4] [r0 r1 c0 c1] ROI
    roinr
    roinc
    
    frmtrk0
    frmtrk1
    nfrmtrk
    
    prnAmat % [roinr,roinc,T]. Amat
    prnTrk % [Tx2]. (x,y) for optimal track using back-propagation. In orig/absolute coords.
    prnTrkAbs %: [Tx2]. (x,y) for track simply following maxima of pbest. In orig/absolute coords.
    prnBest %: [roinr,roinc,T]. pbest(i,j,t) gives the maximum/best probability of reaching (xgrid(i,j),ygrid(i,j)) at time t. [sum of pbest(:,:,t)] ~ 1.
    prnPrev %: [roinr,roinc,T]. pprev(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to pbest(i,j,t).
    prnP0sigs %: [T,2]; % 2D bandwidth
  end
  
  methods
    
    function obj = CPRPrune(imnr,imnc,sigd)
      obj.imnr = imnr;
      obj.imnc = imnc;
      obj.sigD = sigd;
      
      rad = max(imnr,imnc);
      [xgridbig,ygridbig] = meshgrid(-rad:1:rad,-rad:1:rad);
      obj.bigD = obj.computeD(xgridbig,ygridbig,0,0,sigd);
    end
    
    function init(obj,trkPFull,trkPiPt,iPt,frm0,frm1)
      [nfrm,nrep,DD] = size(trkPFull);
      d = 2;
      nptsTracked = DD/d;
      assert(nptsTracked==numel(trkPiPt));
      
      tpf = reshape(trkPFull,nfrm,nrep,nptsTracked,2);
      xytmp = permute(tpf,[2 4 1 3]); % [ndep x 2 x nfrm x nptstrk]

      tf = iPt==trkPiPt;
      assert(nnz(tf)==1);
      xytmp = xytmp(:,:,frm0:frm1,tf); % [nrep x 2 x nfrm];
      
%       FRAMES = 900:1000;

      % FIGURE OUT THE ROI
      ROIPAD = 5;      
      x = squeeze(xytmp(:,1,:));
      y = squeeze(xytmp(:,2,:));
      xmin = floor(min(x(:)));
      xmax = ceil(max(x(:)));
      ymin = floor(min(y(:)));
      ymax = ceil(max(y(:)));
      r0 = max(1,ymin-ROIPAD);
      r1 = min(obj.imnr,ymax+ROIPAD);
      c0 = max(1,xmin-ROIPAD);
      c1 = min(obj.imnc,xmax+ROIPAD);      
      
      rOffset = r0-1;
      cOffset = c0-1;
      % x==c0 will map to x==1
      % y==r0 will map to y==1
      %
      % x==xmin will map to ROIPAD+1 typically; in cases where 
      % xmin<=ROIPAD+1, x==xmin will not change.
      xytmp(:,1,:) = xytmp(:,1,:)-cOffset; 
      xytmp(:,2,:) = xytmp(:,2,:)-rOffset; % y==ymin etc
      
      obj.xy = xytmp;
      obj.roi = [r0 r1 c0 c1];
      obj.roinc = c1-c0+1;
      obj.roinr = r1-r0+1;
      
      x = squeeze(xytmp(:,1,:));
      y = squeeze(xytmp(:,2,:));
      xmin = floor(min(x(:)));
      xmax = ceil(max(x(:)));
      ymin = floor(min(y(:)));
      ymax = ceil(max(y(:)));

      obj.frmtrk0 = frm0;
      obj.frmtrk1 = frm1;
      obj.nfrmtrk = frm1-frm0+1;
      
      fprintf(1,'X: ROI is [%d %d], width=%d. [xmin xmax] is [%d %d].\n',...
        c0,c1,obj.roinc,xmin,xmax);
      fprintf(1,'Y: ROI is [%d %d], height=%d. [ymin ymax] is [%d %d].\n',...
        r0,r1,obj.roinr,ymin,ymax);
      fprintf(1,'Frames to track are: [%d %d], nfrmtrk=%d.\n',...
        frm0,frm1,frm1-frm0+1);
    end
    
    function run(obj)

      nr = obj.roinr;
      nc = obj.roinc;
      T = obj.nfrmtrk;
      
      [xgrid,ygrid] = meshgrid(1:nc,1:nr);
      assert(size(obj.xy,3)==T);
      xy3d = obj.xy(:,:,1:T); % [nRep x 2 x T]
            
      % initialize 
      pbest = nan(nr,nc,T); % pbest(i,j,t) gives the maximum/best probability of reaching (xgrid(i,j),ygrid(i,j)) at time t. [sum of pbest(:,:,t)] ~ 1.
      pprevloc = nan(nr,nc,T); % pprevloc(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to pbest(i,j,t).
      p0sigs = nan(T,2); % 2D bandwidth
      Amatbig = nan(nr,nc,T);
      
      [p0_1,p0sigs(1,:)] = obj.estimateP0(xy3d(:,:,1),xgrid,ygrid);
      pbest(:,:,1) = p0_1;
      pprevloc(:,:,1) = nan; 
      
      % iter
      for t=2:T
        
        [p0_t,p0sigs(t,:)] = obj.estimateP0(xy3d(:,:,t),xgrid,ygrid); % p0_t indexed by r_t
        tmp = sum(p0_t(:));
        fprintf(1,'Sum p0_t is: %.3f\n',tmp);
        Amat = obj.computeAmat(p0_t,xgrid,ygrid); % Amat indexed by r_(t-1)
        p_tm1 = pbest(:,:,t-1); % p_tm1 indexed by r_(t-1)        
        Amatbig(:,:,t) = Amat;
 
        % loop over r_t
        for i=1:nr
        for j=1:nc
          
          % for this pt (x_t,y_t): maximize P_t((x_t,y_t),{observed reps at t} |
          % (x_[t-1],y_[t-1])) * P_[t-1](x_[t-1],y_[t-1]) over (x_[t-1],y_[t-1])
          
          % CHECK THIS WHEN SIGD IS NOT INF
          % Distance weight of (x_[t-1],y_[t-1]) relative to (x_t,y_t)
          % if (x_t,y_t)==(1,1), then Dmat=Dbigmat(nr+1:end-1,nc+1:end-1)
          % if (x_t,y_t)==(2,2), then Dmat=Dbigmat(nr:end-2,nc:end-2)
          % if (x_t,y_t)==(nc,1), then Dmat=Dbigmat(nr+1:end-1,2:nc+1)
          % if (x_t,y_t)==(1,nr), then Dmat=Dbigmat(2:nr+1,nc+1:end-1) 
%           rowStart = nr+2-y_t;
%           colStart = nc+2-x_t;
%           Dmat = Dbigmat(rowStart:rowStart+nr-1,colStart:colStart+nc-1);
          Dmat = obj.fetchD(obj.bigD,i,j,nr,nc);
          assert(isequal(size(Dmat),[nr nc]));
          assert(Dmat(i,j)==1);
          % Dmat is conceptually indexed by x_[t-1],y_[t-1]          
          
          % p_t_and_p_tm1(u,v) is net/joint probability of being
          % at (u,v) at time t-1 and (i,j) at time t. First 3 terms in this
          % product are p_t_given_p_tm1.
          p_t_and_p_tm1 = Amat.*p0_t(i,j).*Dmat.*p_tm1; % indexed by x_[t-1],y_[t-1]
          
          %[pbest(i,j,t),pprevloc(i,j,t)] = max(p_t_and_p_tm1(:));    
          %
          % AL20161206: Don't just take the max; sum over all locs at t-1.
          % However, still use argmax for previous pointer. 
          assert(all(p_t_and_p_tm1(:)>=0));
          pbest(i,j,t) = sum(p_t_and_p_tm1(:));
          [~,pprevloc(i,j,t)] = max(p_t_and_p_tm1(:));
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

      r0 = obj.roi(1);
      c0 = obj.roi(3);
      rOffset = r0-1;
      cOffset = c0-1;
      pTrk(:,1) = pTrk(:,1)+cOffset;
      pTrk(:,2) = pTrk(:,2)+rOffset;
      pTrkAbs(:,1) = pTrkAbs(:,1)+cOffset;
      pTrkAbs(:,2) = pTrkAbs(:,2)+rOffset;    
      obj.prnAmat = Amatbig;
      obj.prnTrk = pTrk;
      obj.prnTrkAbs = pTrkAbs;
      obj.prnBest = pbest;
      obj.prnPrev = pprevloc;
      obj.prnP0sigs = p0sigs;
    end
    
    function compactify(obj)
      obj.prnBest = single(obj.prnBest);
      obj.prnPrev = single(obj.prnPrev);
    end
    
    function Amat = computeAmat(obj,p0,xgrid,ygrid)
      % Compute A assuming (x0,y0) at all grid points
      
      assert(isequal(size(p0),size(xgrid),size(ygrid)));
      
      [nr,nc] = size(p0);
      Amat = nan(size(p0));
      for i=1:nr
      for j=1:nc
        Amat(i,j) = obj.computeA(p0,xgrid,ygrid,i,j);
      end
      end
    end
    
    function A = computeA(obj,p0,xgrid,ygrid,i,j)
      % Compute A, normalization constant for conditional prob
      %
      % p0: [nr x nc] P0 pdf
      % xgrid/ygrid: [nr x nc] x-coord array meshgrid style for p0
      % i, j: row, col indices into xgrid/ygrid for "previous" loc/coord
      %
      % sigD: scalar, SD for distance function
      
      [nr,nc] = size(xgrid);
      %bigDmat = obj.bigD;
      D = obj.fetchD(obj.bigD,i,j,nr,nc);
      %D = CPRPrune.computeD(xgrid,ygrid,x0,y0,sigD);      
      
      prod = p0.*D;      
      A = 1/sum(prod(:));
      % [sum_over_(x,y)_for_given_(x0,y0) A*p*D]==1
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
    
    function Dmat = fetchD(bigDmat,i0,j0,nr,nc)
      % Fetch a D submatrix from a big/full D matrix
      %
      % bigDmat: precomputed big D matrix, square. Center of square
      % corresponds to distance==0, D==1.
      % i0, j0: row, col such that Dmat(i0,j0)==1, or (i0,j0) corresponds 
      % to distance==0 in Dmat submatrix.
      % nr, nc: nrows, ncols of resulting Dmat submatrix.
      %
      % Dmat: [nrxnc] submatrix of bigDmat where Dmat(i0,j0)==1.
      
      [nrbig,ncbig] = size(bigDmat);
      assert(nrbig==ncbig);
      ibigctr = (nrbig+1)/2;
      assert(bigDmat(ibigctr,ibigctr)==1);
      
      rowOffset = ibigctr-i0+1; % when i0==1,rowIdxs starts at ibigctr. when i0==nr, rowIdxs ends at ibigctr
      colOffset = ibigctr-j0+1;
      rowIdxs = rowOffset:rowOffset+nr-1;
      colIdxs = colOffset:colOffset+nc-1;
      
      Dmat = bigDmat(rowIdxs,colIdxs);
      assert(isequal(size(Dmat),[nr nc]));
      assert(Dmat(i0,j0)==1);
    end
  end
      
end