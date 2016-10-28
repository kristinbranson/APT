classdef RegressorCascade < handle
  
  properties
    % model/params
    prmModel 
    prmTrainInit
    prmReg 
    prmFtr 
  end
  
  properties
    pGTNTrn % [NtrnxD] normalized shapes used during most recent full training
    
    ftrSpecs % [nMjr] cell array of feature definitions/specifications. ftrSpecs{i} is either [], or a struct specifying F features
    %ftrs % [nMjr] cell array of instantiated features. ftrs{i} is either [], or NxF    
    ftrsUse % [nMjr x nMnr x M x nUse] selected feature subsets (M=fern depth). ftrsUse(iMjr,iMnr,:,:) contains selected features for given iteration. nUse is 1 by default or can equal 2 for ftr.metatype='diff'.        
    
    fernN % [nMjr x nMnr] total number of data points run through fern regression 
    fernMu % [nMjr x nMnr x D] mean output (Y) encountered by fern regression
    fernThresh % [nMjr x nMnr x M] fern thresholds
    fernCounts % [nMjr x nMnr x 2^M X D] count of number of shapes binned for each coord, treating NaNs as missing data
    fernSums % [nMjr x nMnr x 2^M X D] sum of dys for each bin/coord, treating NaNs as missing data
    fernOutput % [nMjr x nMnr x 2^M x D] output/shape correction for each fern bin
    fernTS % [nMjr x nMnr] timestamp last mod .fernCounts/Output
    
    trnLog % struct array, one el per train/retrain action
  end
  properties (Dependent)
    nMajor
    nMinor
    M
    mdld
    mdlD
    hasTrained % scalar logical; true if at least one full training has occurred
  end
  
  methods
    function v = get.nMajor(obj)
      v = obj.prmReg.T;
    end
    function v = get.nMinor(obj)
      v = obj.prmReg.K;
    end
    function v = get.M(obj)
      v = obj.prmReg.M;
    end
    function v = get.mdld(obj)
      v = obj.prmModel.d;
    end
    function v = get.mdlD(obj)
      v = obj.prmModel.D;
    end
    function v = get.hasTrained(obj)
      v = ~isempty(obj.trnLogMostRecentTrain());
    end
  end
  
  methods
    
    function obj = RegressorCascade(sPrm)
      % sPrm: parameter struct
      
      if isfield(sPrm.Model,'D')
        assert(sPrm.Model.D==sPrm.Model.d*sPrm.Model.nfids);
      else
        sPrm.Model.D = sPrm.Model.d*sPrm.Model.nfids;
      end
      obj.prmModel = sPrm.Model;
      obj.prmTrainInit = sPrm.TrainInit;
      obj.prmReg = sPrm.Reg;
      obj.prmFtr = sPrm.Ftr;  
      obj.init();
    end
    
    function init(obj,varargin)
      % Clear/init everything but mdl/params
      
      initTrnLog = myparse(varargin,...
        'initTrnLog',true); 
      
      obj.pGTNTrn = [];
      
      nMjr = obj.nMajor;
      nMnr = obj.nMinor;
      MM = obj.M;
      ftrMetaType = obj.prmFtr.metatype;
      switch ftrMetaType
        case 'single'
          nUse = 1;
        case 'diff'
          nUse = 2;
        otherwise
          assert(false);
      end      
      
      obj.ftrSpecs = cell(nMjr,1);
      obj.ftrsUse = nan(nMjr,nMnr,MM,nUse);
      
      obj.fernN = zeros(nMjr,nMnr);
      obj.fernMu = nan(nMjr,nMnr,obj.mdlD);    
      obj.fernThresh = nan(nMjr,nMnr,MM);
      obj.fernSums = zeros(nMjr,nMnr,2^MM,obj.mdlD);
      obj.fernCounts = zeros(nMjr,nMnr,2^MM,obj.mdlD);
      obj.fernOutput = nan(nMjr,nMnr,2^MM,obj.mdlD);
      obj.fernTS = -inf*ones(nMjr,nMnr);
      
      if initTrnLog
        obj.trnLogInit();
      end
    end
    
    % Notes on shapes, views, coords
    %
    % Pixel-coordinates on actual images are called "image coords". These
    % are eg (row,col) for array-indexing or (x=col,y=row) for xy-style
    % indexing. Image coords are defined per-, or with-respect-to-, a view. 
    % A 3d-shape can be projected onto any/each view to generate image 
    % coords.
    %
    % For 2D models, p-shapes in absolute coords are the same as
    % (x=col,y=row) image coordinates.
    %
    % For 3D models, p-shapes in absolute coords are 3D coords in the
    % camera coord system of model.Prm3D.iViewBase. p-shapes usually come
    % in rows of length D, so that a set of p-shapes is [NxD]. Here,
    % D=nfids*d, and the format of p is [x1 x2 x3... x_nfids y1 y2 y3 ...
    % y_nfids z1 z2 z3 ... z_nfids]. model.Prm3D.calrig is a CalRig that
    % knows how to i) go from the camera coord sys of view A to view B; and
    % ii) knows how to go from the camera coord sys of a given view to 
    % image coords for that view.
    
    % Note on bboxes. Bounding boxes are for allowing tracking videos where
    % targets are in different locations and/or at different scales in the
    % image.
    %
    % In a 2D video, bboxes will be 2D ROIs centered on targets and sized
    % appropriately for the target (apparent) size. During training, shapes
    % in absolute coords are "projected" onto the bounding box, ie
    % coordinates are normalized relative to the bounding box into the
    % range [-1,1] with -1 representing the left/lower edge and 1
    % representing the right/upper edge of the box.
    %
    % For 2D videos with a single target, bboxes will often just be the
    % entire image.
    %
    % For 3D videos, the shapes are in camera coords of the "base" view.
    % BBoxes are therefore 3D ROIs in the base camera coord system.
    
    
    
    %#3DOK
    function [ftrs,iFtrs] = computeFeatures(obj,t,I,bboxes,p,pIidx,tfused) % obj const
      % t: major iteration
      % I: [NxnView] Cell array of images (nView==1) or imageSets (nView>1)
      % bboxes: [Nx2*d]. Currently unused (used only for occlusion)
      % p: [QxD] shapes, absolute coords. p(i,:) is p-vector for image(s) I(pIidx(i),:)
      % pIidx: [Q] indices into rows of I (imageSets) labeling 1st dim of p
      % tfused: if true, only compute those features used in obj.ftrsUse(t,:,:,:)
      %
      % ftrs: If tfused==false, then [QxF]; otherwise [QxnUsed]
      % iFtrs: feature indices labeling cols of ftrs
      
      fspec = obj.ftrSpecs{t};

      if tfused
        iFtrs = obj.ftrsUse(t,:,:,:);
        iFtrs = unique(iFtrs(:));
      else
        iFtrs = 1:fspec.F;
        iFtrs = iFtrs(:);
      end        
      
      assert(~isempty(fspec),'No feature specifications for major iteration %d.',t);
      switch fspec.type
        case 'kborig_hack'
          assert(obj.prmModel.d==2,'2D only at the moment.');
          assert(~tfused,'Unsupported.');
          ftrs = shapeGt('ftrsCompKBOrig',obj.prmModel,p,I,fspec,...
            pIidx,[],bboxes,obj.prmReg.occlPrm);
        case {'1lm' '2lm' '2lmdiff'}
          fspec = rmfield(fspec,'pids');
          fspec.F = numel(iFtrs);
          fspec.xs = fspec.xs(iFtrs,:);
          ftrs = shapeGt('ftrsCompDup2',obj.prmModel,p,I,fspec,...
            pIidx,[],bboxes,obj.prmReg.occlPrm);
        otherwise
          assert(false,'Unrecognized feature specification type.');
      end
    end
    
    %#3DOK
    function [pAll,pIidx] = trainWithRandInit(obj,I,bboxes,pGT,varargin)
      % I: [NxnView] cell array of images
      % bboxes: [Nx2*d]
      % pGT: [NxD] GT labels (absolute coords)
      %
      % pAll: [(N*Naug)xDx(T+1)] propagated training shapes (absolute coords)
      % pIidx: [N*Naug] indices into I labeling rows of pAll
      %
      % Initialization notes. Two sets of shapes to draw from for
      % initialization. If initpGTNTrn, use the set .pGTNTrn; otherwise,
      % use pGT. Typically, initpGTNTrn would be used for incremental
      % (re)trains, where .pGTNTrn is set and pGT is small/limited.
      % Meanwhile, pGT would be used on first/fresh trains, where .pGTNTrn
      % may not be populated and pGT is large.
      %
      % In drawing from a set shape distribution, we are biasing towards
      % the most/more common shapes. However, we also jitter, so that may
      % be okay.
      
      [initpGTNTrn,loArgs] = myparse_nocheck(varargin,...
        'initpGTNTrn',false... % if true, init with .pGTNTrn rather than pGT
        );
      
      model = obj.prmModel;
      tiPrm = obj.prmTrainInit;
      Naug = tiPrm.Naug;  
      if initpGTNTrn
        pNInitSet = obj.pGTNTrn;
        selfSample = false;
      else % init from pGt
        pNInitSet = shapeGt('projectPose',model,pGT,bboxes);
        selfSample = true;
      end
      p0 = Shape.randInitShapes(pNInitSet,Naug,model,bboxes,...
        'dorotate',tiPrm.augrotate,...
        'bboxJitterfac',tiPrm.augjitterfac,...
        'selfSample',selfSample);
      N = size(I,1);
      szassert(p0,[N Naug model.D]);
      
      p0 = reshape(p0,[N*Naug model.D]);
      pIidx = repmat(1:N,[1 Naug])';
      pAll = obj.train(I,bboxes,pGT,p0,pIidx,loArgs{:});
    end    
    
    %#3DOK
    function pAll = train(obj,I,bboxes,pGT,p0,pIidx,varargin)
      %
      % I: [NxnView] cell array of images
      % bboxes: [Nx2*d]
      % pGT: [NxD] GT labels (absolute coords)
      % p0: [QxD] initial shapes (absolute coords).
      % pIidx: [Q] indices into I for p0
      %
      % pAll: [QxDxT+1] propagated training shapes (absolute coords)
      
      [verbose,hWB,update] = myparse(varargin,...
        'verbose',1,...
        'hWaitBar',[],...
        'update',false... % if true, incremental update
        );
      
      model = obj.prmModel;
      
      [NI,nview] = size(I);
      assert(nview==model.nviews);
      assert(isequal(size(bboxes),[NI 2*obj.mdld]));
      assert(isequal(size(pGT),[NI obj.mdlD]));      
      [Q,D] = size(p0);
      assert(D==obj.mdlD);
      assert(numel(pIidx)==Q);
      
      if update && ~obj.hasTrained
        error('RegressorCascade:noTrain',...
          'Cannot perform incremental train without first doing a full train.');
      end

      pGTFull = pGT(pIidx,:);
      T = obj.nMajor;
      pAll = zeros(Q,D,T+1);
      pAll(:,:,1) = p0;
      t0 = 1;
      pCur = p0;
      bboxesFull = bboxes(pIidx,:);
      
      if ~update
        obj.init('initTrnLog',false);
        % record normalized training shapes for propagation initialization
        pGTN = shapeGt('projectPose',model,pGT,bboxes);
        obj.pGTNTrn = pGTN;
      end
      
      loss = mean(shapeGt('dist',model,pCur,pGTFull));
      if verbose
        fprintf('  t=%i/%i       loss=%f     \n',t0-1,T,loss);
      end
      tStart = clock;
      
      paramFtr = obj.prmFtr;
      ftrRadiusOrig = paramFtr.radius; % for t-dependent ftr radius
      paramReg = obj.prmReg;
      
      maxFernAbsDeltaPct = nan(1,T);
      for t=t0:T
        if paramReg.USE_AL_CORRECTION
          assert(model.d==2,'Currently supported only for d==2.');
          pCurN_al = shapeGt('projectPose',model,pCur,bboxesFull);
          pGtN_al = shapeGt('projectPose',model,pGTFull,bboxesFull);
          assert(isequal(size(pCurN_al),size(pGtN_al)));
          pDiffN_al = Shape.rotInvariantDiff(pCurN_al,pGtN_al,1,3); % XXXAL HARDCODED HEAD/TAIL
          pTar = pDiffN_al;
        else
          pTar = shapeGt('inverse',model,pCur,bboxesFull); % pCur: absolute. pTar: normalized
          pTar = shapeGt('compose',model,pTar,pGTFull,bboxesFull); % pTar: normalized
        end
        
        if numel(ftrRadiusOrig)>1
          paramFtr.radius = ftrRadiusOrig(min(t,numel(ftrRadiusOrig)));
        end
        
        % Generate feature specs
        if ~update
          switch paramFtr.type
            case {'kborig_hack'}
              fspec = shapeGt('ftrsGenKBOrig',model,paramFtr);
            case {'1lm' '2lm' '2lmdiff'}
              fspec = shapeGt('ftrsGenDup2',model,paramFtr);
          end
          obj.ftrSpecs{t} = fspec;
        end
        
        % compute features for current training shapes
        [X,iFtrsComp] = obj.computeFeatures(t,I,bboxes,pCur,pIidx,update);
        
        % Regress
        paramReg.ftrPrm = paramFtr;
        paramReg.prm.useFern3 = true;
        fernOutput0 = squeeze(obj.fernOutput(t,:,:,:));
        if ~update
          [regInfo,pDel] = regTrain(X,pTar,paramReg);
          assert(iscell(regInfo) && numel(regInfo)==obj.nMinor);
          for u=1:obj.nMinor
            ri = regInfo{u};          
            obj.ftrsUse(t,u,:,:) = ri.fids';          
            obj.fernN(t,u) = ri.N;
            obj.fernMu(t,u,:) = ri.yMu;
            obj.fernThresh(t,u,:) = ri.thrs;
            obj.fernSums(t,u,:,:) = ri.fernSum;
            obj.fernCounts(t,u,:,:) = ri.fernCount;
            obj.fernOutput(t,u,:,:) = ri.ysFern;
            obj.fernTS(t,u) = now();
          end
        else
          % update: fernN, fernCounts, fernSums, fernOutput, fernTS
          % calc: pDel
          
          assert(obj.mdld==2,'Not checked mdl.d~=2.');
          pDel = obj.fernUpdate(t,X,iFtrsComp,pTar,paramReg);
        end
        fernOutput1 = squeeze(obj.fernOutput(t,:,:,:));
        maxFernAbsDeltaPct(t) = obj.computeMaxFernAbsDelta(fernOutput0,fernOutput1);
                  
        % Apply pDel
        if paramReg.USE_AL_CORRECTION
          assert(obj.mdld==2,'Unchecked for 3D.');
          pCur = Shape.applyRIDiff(pCurN_al,pDel,1,3); %XXXAL HARDCODED HEAD/TAIL
          pCur = shapeGt('reprojectPose',model,pCur,bboxesFull);
        else
          pCur = shapeGt('compose',model,pDel,pCur,bboxesFull);
          pCur = shapeGt('reprojectPose',model,pCur,bboxesFull);
        end
        pAll(:,:,t+1) = pCur;
        
        errPerEx = shapeGt('dist',model,pCur,pGTFull);
        loss = mean(errPerEx);        
        if verbose
          msg = tStatus(tStart,t,T);
          fprintf(['  t=%i/%i       loss=%f     ' msg],t,T,loss);
        end        
      end
      
      if update
        act = 'update';
      else
        act = 'train';
      end      
      obj.trnLog(end+1,1).action = act;
      obj.trnLog(end).ts = now();
      obj.trnLog(end).nShape = Q;
      obj.trnLog(end).maxFernAbsDeltaPct = maxFernAbsDeltaPct;
    end
    
    function trnLogInit(obj)
      obj.trnLog = struct(...
        'action',cell(0,1),... % 'train' or 'retrain'
        'ts',[],... % timestamp
        'nShape',[],... % number of shapes (after any augmentation) trained/added
        'maxFernAbsDeltaPct',[]... % 1xnMjr; maximum delta (L2 norm, pct of mu) in obj.fernOutput(iMjr,:,:,:) over all
        );                         % minor iters, fern bins
    end
        
    function iTL = trnLogMostRecentTrain(obj)
      tl = obj.trnLog;
      act = {tl.action};
      iTL = find(strcmp(act,'train'),1,'last');
    end
    
    function trnLogPrintSinceLastTrain(obj)
      % Pretty-print log from last (full) train onwards

      iTL = obj.trnLogMostRecentTrain();
      if isempty(iTL)
        fprintf('No training has occurred.\n');
      else
        tl = obj.trnLog;
        for i=iTL:numel(tl)
          tlcurr = tl(i);
          fprintf('%s: %s with nShape=%d\n',...
            datestr(tlcurr.ts,'mmm-dd HH:MM:SS'),...
            tlcurr.action,tlcurr.nShape);
        end
      end      
    end
       
    %#3DOK
    function p_t = propagate(obj,I,bboxes,p0,pIidx,varargin) % obj const
      % Propagate shapes through regressor cascade.
      %
      % I: [NxnView] Cell array of images
      % bboxes: [Nx2*d]
      % p0: [QxD] initial shapes, absolute coords, eg Q=N*augFactor
      % pIidx: [Q] indices into (rows of) I for rows of p0
      %
      % p_t: [QxDx(T+1)] All shapes over time. p_t(:,:,1)=p0; p_t(:,:,end)
      % is shape after T'th major iteration.
         
      [t0,hWB] = myparse(varargin,...
        't0',1,... % initial/starting major iteration
        'hWaitBar',[]);
      tfWB = ~isempty(hWB);
    
      model = obj.prmModel;

      [NI,nview] = size(I);
      assert(nview==model.nviews);
      assert(isequal(size(bboxes),[NI 2*obj.mdld]));
      [Q,D] = size(p0);
      assert(numel(pIidx)==Q && all(ismember(pIidx,1:NI)));
      assert(D==obj.mdlD);
  
      ftrMetaType = obj.prmFtr.metatype;
      bbs = bboxes(pIidx,:);
      T = obj.nMajor;
      p_t = zeros(Q,D,T+1); % shapes over all initial conds/iterations, absolute coords
      p_t(:,:,1) = p0;
      p = p0; % current/working shape, absolute coords
                   
      if tfWB
        waitbar(0,hWB,'Applying cascaded regressor');
      end
      for t = t0:T
        if tfWB
          waitbar(t/T,hWB);
        else
          fprintf(1,'Applying cascaded regressor: %d/%d\n',t,T);
        end
              
        [X,iFtrsComp] = obj.computeFeatures(t,I,bboxes,p,pIidx,true);
        assert(numel(iFtrsComp)==size(X,2));

        % Compute shape correction (normalized units) by summing over
        % microregressors
        pDel = zeros(Q,D);
        for u=1:obj.nMinor
          x = obj.computeMetaFeature(X,iFtrsComp,t,u,ftrMetaType);
          thrs = squeeze(obj.fernThresh(t,u,:));
          inds = fernsInds(x,uint32(1:obj.M),thrs(:)'); 
          yFern = squeeze(obj.fernOutput(t,u,inds,:));
          assert(ndims(yFern)==2); %#ok<ISMAT>
          pDel = pDel + yFern; % normalized units
        end
        
        if obj.prmReg.USE_AL_CORRECTION
          assert(model.d==2,'Unchecked 3d');
          p1 = shapeGt('projectPose',model,p,bbs); % p1 is normalized        
          p = Shape.applyRIDiff(p1,pDel,1,3); % XXXAL HARDCODED HEAD/TAIL
        else
          p = shapeGt('compose',model,pDel,p,bbs); % p (output) is normalized
        end
        p = shapeGt('reprojectPose',model,p,bbs); % back to absolute coords
        p_t(:,:,t+1) = p;
      end
    end
    
    %#3DOK
    function [p_t,pIidx,p0info] = propagateRandInit(obj,I,bboxes,prmTestInit,varargin) % obj const
      % Wrapper for propagate(), randomly init replicate cloud from
      % obj.pGTNTrn
      %
      % p_t: [QxDx(T+1)] All shapes over time. p_t(:,:,1)=p0; p_t(:,:,end)
      % is shape after T'th major iteration.
      % pIidx: labels for rows of p_t, indices into I
      % p0info: struct containing initial shape info
      
      model = obj.prmModel;
      [N,nview] = size(I);
      assert(nview==model.nviews);
      szassert(bboxes,[N 2*model.d]);
      
      Naug = prmTestInit.Nrep;
      pNInitSet = obj.pGTNTrn;
      [p0,p0info] = Shape.randInitShapes(pNInitSet,Naug,model,bboxes,...
        'dorotate',prmTestInit.augrotate,...
        'bboxJitterfac',prmTestInit.augjitterfac,...
        'selfSample',false);
      szassert(p0,[N Naug model.D]);
      p0info.p0_1 = squeeze(p0(1,:,:)); % absolute coords
      p0info.bbox1 = bboxes(1,:);
      
      p0 = reshape(p0,[N*Naug model.D]);
      pIidx = repmat(1:N,[1 Naug])';
      p_t = obj.propagate(I,bboxes,p0,pIidx,varargin{:});      
    end
    
    %# XXX TODO3D
    function yPred = fernUpdate(obj,t,X,iFtrsComp,yTar,prmReg)
      % Incremental update of fern structures
      %
      % t: major iter
      % X: [QxnUsed], computed features
      % iFtrsComp: [nUsed], feature indices labeling cols of X
      % yTar: [QxD] target shapes
      % prmReg: 
      %
      % yPred: [QxD], fern prediction on X (summed/boosted over minor iters)
      
      [Q,nU] = size(X);
      assert(numel(iFtrsComp)==nU);
      D = obj.mdlD;
      assert(isequal(size(yTar),[Q D]));
      
      ftrMetaType = obj.prmFtr.metatype;
      MM = obj.M;
      fids = uint32(1:MM);
      ySum = zeros(Q,D); % running accumulation of approx to pTar
      for u=1:obj.nMinor
        yTarMnr = yTar - ySum;
        x = obj.computeMetaFeature(X,iFtrsComp,t,u,ftrMetaType);
        assert(isequal(size(x),[Q MM]));
        thrs = squeeze(obj.fernThresh(t,u,:));
        
        yMuOrig = reshape(obj.fernMu(t,u,:),[1 D]);
        dY = bsxfun(@minus,yTarMnr,yMuOrig);
        [inds,dyFernSum,~,dyFernCnt] = Ferns.fernsInds3(x,fids,thrs,dY);
        indsTMP = fernsInds(x,fids,thrs); % dumb check
        assert(isequal(inds,indsTMP));
               
        obj.fernN(t,u) = obj.fernN(t,u) + Q;
        obj.fernCounts(t,u,:,:) = squeeze(obj.fernCounts(t,u,:,:)) + dyFernCnt;
        obj.fernSums(t,u,:,:) = squeeze(obj.fernSums(t,u,:,:)) + dyFernSum;
        
        counts = squeeze(obj.fernCounts(t,u,:,:));
        sums = squeeze(obj.fernSums(t,u,:,:));
        ysFernCntUse = max(counts+prmReg.prm.reg*obj.fernN(t,u),eps);
        ysFern = bsxfun(@plus,sums./ysFernCntUse,yMuOrig);
        
        obj.fernOutput(t,u,:,:) = ysFern;
        
        yPredMnr = ysFern(inds,:);
        ySum = ySum + yPredMnr;
        
        obj.fernTS(t,u) = now();
      end
      
      yPred = ySum;

      
      % See train() for parameter descriptions. The difference here is that
      % I, bboxes, pGT etc represent new/additional images+gtShapes to
      % integrate into the cascade.
      %
      % In an incremental update:
      % - The set of available features for each major iter are re-used;
      % - The selected subset of features used in each minor iter are re-used;
      % - fern thresholds are not touched;
      % -.fernN will be updated
      % - .fernCounts will be augmented with ~<Q counts for each coord, for
      %   each (mjr,mnr) iteration;
      % - .fernOutput will be updated with adding in a contribution for Q
      %   new datapoints (weighted against the N_existing datapoints)
      % - The new shapes are propagated through the cascade using the 
      %   updated fernCounts/Output etc. Note however that the previous/
      %   existing datapoints are NOT repropagated using these updated fern
      %   props. This is a fundamental deviation from a full retraining.
      
    end
    
        % ----- BELOW HERE NOT SURE TODO3D -----

    %#3DOK
    function x = computeMetaFeature(obj,X,iFtrsX,t,u,metatype)
      % Helper function to compute meta-features
      %
      % X: [QxZ] computed features
      % iFtrsX: [Z] feature indices labeling cols of X (indices into ftrsSpecs{t})
      % t: major iter
      % u: minor iter
      % metatype: either 'single' or 'diff'
      %
      % x: [QxM] meta-features
      
      iFtrsUsed = squeeze(obj.ftrsUse(t,u,:,:)); % [MxnUse]
      nUse = size(iFtrsUsed,2);
      switch metatype
        case 'single'
          assert(nUse==1);
          [~,loc] = ismember(iFtrsUsed,iFtrsX);
          x = X(:,loc);
        case 'diff'
          assert(nUse==2);
          [~,loc1] = ismember(iFtrsUsed(:,1),iFtrsX);
          [~,loc2] = ismember(iFtrsUsed(:,2),iFtrsX);
          x = X(:,loc1)-X(:,loc2);
      end     
    end
    
    %#3DOK
    function maxFernAbsDeltaPct = computeMaxFernAbsDelta(obj,fernOutput0,fernOutput1)
      % fernOutput0/1: [nMnr x 2^M x D]
      %
      % maxFernAbsDeltaPct: scalar. Maximum of L2Delta./L2mu, over
      % all bins, over all minor iters
      
      assert(isequal(obj.nMinor,size(fernOutput0,1),size(fernOutput1,1)));
      
      maxFernAbsDeltaPct = nan(obj.nMinor,1);
      for iMnr = 1:obj.nMinor
        del = squeeze(fernOutput1(iMnr,:,:)-fernOutput0(iMnr,:,:)); % [2^M x D]
        del = sqrt(sum(del.^2,2)); % [2^Mx1], L2 deviation for each fern bin (for this minor iter)
        mu = squeeze(fernOutput1(iMnr,:,:)+fernOutput0(iMnr,:,:))/2;
        mu = sqrt(sum(mu.^2,2)); % [2^Mx1], L2 of fern output vec for each fern bin (etc)
        
        maxFernAbsDeltaPct(iMnr) = max(del./mu);
      end
      
      maxFernAbsDeltaPct = max(maxFernAbsDeltaPct);      
    end
    
    function setPrm(obj,sPrm)
      % stupidity

      obj.prmModel = sPrm.Model;
      obj.prmTrainInit = sPrm.TrainInit;
      obj.prmReg = sPrm.Reg;
      obj.prmFtr = sPrm.Ftr;      
    end
    
  end
  
  methods (Static) % utils
    
    function hFig = createP0DiagImg(I,p0info)
      % Visualize initial random shapes
      %
      % I: [NxnView] cell array of images
      % p0Info: struct containing initial shape randomization info, see eg
      %   propagateRandInit()
      
      model = p0info.model;
      assert(model.d==2,'Unsupported for d~=2.');
      
      hFig = figure;
      ax = axes;
      im = I{1}(:,:,1);
      imagesc([-1 1],[-1 1],im,'parent',ax);
      truesize(hFig);
      colormap(ax,'gray');
      hold(ax,'on');
      p0_1 = p0info.p0_1;
      bbox1 = p0info.bbox1;
      p0_1N = shapeGt('projectPose',model,p0_1,repmat(bbox1,size(p0_1,1),1));
      pNmu = p0info.pNmu;
      [Naug,D] = size(p0_1); 
      szassert(pNmu,[1 D]);
      
      npts = D/p0info.model.d;
      colors = lines(npts);
      for ipt=1:npts
        clr = colors(ipt,:);
        plot(ax,p0_1N(:,ipt),p0_1N(:,ipt+npts),'.','Color',clr,'MarkerSize',10); % plot all replicates for ipt
        plot(ax,pNmu(ipt),pNmu(ipt+npts),'ws','MarkerFaceColor',clr*.75+.25);
      end
      tstr = sprintf('naug:%d. npN:%d. doRot:%d. jitterfac:%d.',...
        Naug,p0info.npN,p0info.doRotate,p0info.bboxJitterFac);
      title(ax,tstr,'interpreter','none','fontweight','bold');
      hFig.UserData = p0info;
    end
    
  end
    
end
  
function msg = tStatus(tStart,t,T)
elptime = etime(clock,tStart);
fracDone = max( t/T, .00001 );
esttime = elptime/fracDone - elptime;
if( elptime/fracDone < 600 )
  elptimeS  = num2str(elptime,'%.1f');
  esttimeS  = num2str(esttime,'%.1f');
  timetypeS = 's';
else
  elptimeS  = num2str(elptime/60,'%.1f');
  esttimeS  = num2str(esttime/60,'%.1f');
  timetypeS = 'm';
end
msg = ['[elapsed=' elptimeS timetypeS ...
  ' / remaining~=' esttimeS timetypeS ']\n' ];
end