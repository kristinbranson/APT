classdef CPRPruneGen < handle
  
  properties
    imnr % number rows image
    imnc % number cols image
    sigD % sigma for transition/position distance function (Dmat)
    bigD % precomputed LOG big D matrix 
    lam % score is computed as appearancescore + lam*positionScore

    xy % [nRep x 2 x nFrm] shifted xy coords for pt of interest, all replicates/frames 
    roi % [1x4] [r0 r1 c0 c1] ROI
    roinr
    roinc
    
    frmtrk0
    frmtrk1
    nfrmtrk
    
    prnBest %: [roinr,roinc,T]. pbest(i,j,t) gives the maximum/best probability of reaching (xgrid(i,j),ygrid(i,j)) at time t. [sum of pbest(:,:,t)] ~ 1.
    prnPrev %: [roinr,roinc,T]. pprev(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to pbest(i,j,t).
    prnAppScore % [roinr,roinc,T].
    prnTrk % [Tx2]. (x,y) for optimal track using back-propagation. In orig/absolute coords.
    prnTrkAbs %: [Tx2]. (x,y) for track simply following maxima of pbest. In orig/absolute coords.
    prnP0sigs %: [T,2]; % 2D bandwidth
  end
  
  methods
    
    function obj = CPRPruneGen(imnr,imnc,sigd,lam)
      obj.imnr = imnr;
      obj.imnc = imnc;
      obj.sigD = sigd;
      obj.lam = lam;
      
      rad = max(imnr,imnc);
      [xgridbig,ygridbig] = meshgrid(-rad:1:rad,-rad:1:rad);
      obj.bigD = log(obj.computeD(xgridbig,ygridbig,0,0,sigd));
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
      lambda = obj.lam;
      
      [xgrid,ygrid] = meshgrid(1:nc,1:nr);
      assert(size(obj.xy,3)==T);      
            
      % initialize 
      scoreapp = nan(nr,nc,T); % appearance score
      scorebest = nan(nr,nc,T); % scorebest(i,j,t) gives the maximum/best probability/score of reaching (xgrid(i,j),ygrid(i,j)) at time t. bigger is more likely
      prevloc = nan(nr,nc,T); % prevloc(i,j,t) is a linear index into the image (or xgrid/ygrid) giving the previous location (at t-1) leading to scorebest(i,j,t).
      psigs = nan(T,2); % 2D kde bandwidth
      
      % Given scorebest(:,:,t):
      % 1. Compute (t+1) Appearance Score on x/ygrid as log(kde(replicates at time t+1))
      % 2. For each gridpt, compute transition cost from prev (t) over all prev gridpts, then
      %   a. Compute (t+1) total score as AppScore + Best_over_all_prev_gridpts(prevscore + transitioncost)
      %   b. Record best score in scorebest(i,j,t+1)
      %   c. Record best/most likely prevpt in prevloc(i,j,t+1)

      t = 1;
      [p0,psigs(t,:)] = obj.estimateP0(obj.xy(:,:,t),xgrid,ygrid);
      scorebest(:,:,t) = log(p0);
      prevloc(:,:,t) = nan; 
      scoreapp(:,:,t) = log(p0);
            
      INLINEBIGD = false;
      if INLINEBIGD
        bigDmat = obj.bigD;
        [nrbig,ncbig] = size(bigDmat);
        assert(nrbig==ncbig);
        ibigctr = (nrbig+1)/2;
        assert(bigDmat(ibigctr,ibigctr)==0); % bigDmat is now log(D)
      end
      
      for t=2:T
        [p0,psigs(t,:)] = obj.estimateP0(obj.xy(:,:,t),xgrid,ygrid); % p0_t indexed by r_t
        fprintf(1,'Sum p0_t is: %.5f\n',sum(p0(:))); % sanity, should very nearly equal 1; if not, KDE extends beyond roi/grid
        logp0 = log(p0);
        scoreapp(:,:,t) = logp0;
        scoreprev = scorebest(:,:,t-1); % indexed by r_(t-1)
       
        % loop over r_t
        for i=1:nr
        for j=1:nc      
          if INLINEBIGD
            rowOffset = ibigctr-i+1; % when i0==1,rowIdxs starts at ibigctr. when i0==nr, rowIdxs ends at ibigctr
            colOffset = ibigctr-j+1;
            rowIdxs = rowOffset:rowOffset+nr-1;
            colIdxs = colOffset:colOffset+nc-1;            
            logDmat = bigDmat(rowIdxs,colIdxs);
          else
            logDmat = obj.fetchD(obj.bigD,i,j,nr,nc);
          end
          szassert(logDmat,[nr nc]);
          assert(logDmat(i,j)==0);
          
          scoretransition = lambda * logDmat; % indexed by r_(t-1)
          scoretot = logp0(i,j) + scoretransition + scoreprev;
          [scorebest(i,j,t),prevloc(i,j,t)] = max(scoretot(:));          
        end
        end
                
        fprintf('T=%d. psigs=%s.\n',t,mat2str(psigs(t,:),3));
      end
      
      % Find tracks
      pTrk = nan(T,2);
      pTrkAbs = nan(T,2);
      for t = T:-1:1
        pbestCurr = scorebest(:,:,t);

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
          idxPrev = prevloc(rt,ct,t);
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

      obj.prnBest = scorebest;
      obj.prnPrev = prevloc;
      obj.prnAppScore = scoreapp;
      obj.prnTrk = pTrk;
      obj.prnTrkAbs = pTrkAbs;
      obj.prnP0sigs = psigs;
    end
    
    function compactify(obj)
      obj.prnBest = single(obj.prnBest);
      obj.prnPrev = single(obj.prnPrev);
      obj.prnAppScore = single(obj.prnAppScore);
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
      assert(bigDmat(ibigctr,ibigctr)==0); % bigDmat is now log(D)
      
      rowOffset = ibigctr-i0+1; % when i0==1,rowIdxs starts at ibigctr. when i0==nr, rowIdxs ends at ibigctr
      colOffset = ibigctr-j0+1;
      rowIdxs = rowOffset:rowOffset+nr-1;
      colIdxs = colOffset:colOffset+nc-1;
      
      Dmat = bigDmat(rowIdxs,colIdxs);
      assert(isequal(size(Dmat),[nr nc]));
      assert(Dmat(i0,j0)==0);
    end
  end
      
end