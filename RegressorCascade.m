classdef RegressorCascade < handle
  
  properties
    mdl % model
    regPrm % parameters    
    ftrPrm % 
    
    ftrSpecs % [nMjr] cell array of feature definitions/specifications. ftrSpecs{i} is either [], or a struct specifying F features
    %ftrs % [nMjr] cell array of instantiated features. ftrs{i} is either [], or NxF    
    ftrsUse % [nMjr x nMnr x M x nUse] selected feature subsets (M=fern depth). ftrsUse(iMjr,iMnr,:,:) contains selected features for given iteration. nUse is 1 by default or can equal 2 for ftr.metatype='diff'.    
    
    fernThresh % [nMjr x nMnr x M] fern thresholds
    fernCounts % [nMjr x nMnr x 2^M] number of shapes binned
    fernOutput % [nMjr x nMnr x 2^M x D] output/shape correction for each fern bin
    fernTS % [nMjr x nMnr] timestamp last mod .fernCounts/Output     
  end
  properties (Dependent)
    nMajor
    nMinor
    M
    mdld
    mdlD
  end
  
  methods 
    function v = get.nMajor(obj)
      v = obj.regPrm.T;
    end
    function v = get.nMinor(obj)
      v = obj.regPrm.K;
    end
    function v = get.M(obj)
      v = obj.regPrm.M;
    end
    function v = get.mdld(obj)
      v = obj.mdl.d;
    end
    function v = get.mdlD(obj)
      v = obj.mdl.nfids & obj.mdl.d;
    end
  end
  
  methods
    
    function obj = RegressorCascade(model,regPrm,ftrPrm)
      obj.mdl = model;
      obj.regPrm = regPrm;
      obj.ftrPrm = ftrPrm;      
      obj.init();
    end
    
    function init(obj)
      % init .ftr*, .fern* 
      
      nMjr = obj.nMajor;
      nMnr = obj.nMinor;
      MM = obj.M;
      ftrMetaType = obj.ftrPrm.metatype;
      
      switch ftrMetaType
        case 'single'
          nUse = 1;
        case 'diff'
          nUse = 2;
        otherwise
          assert(false);
      end      
      
      obj.ftrSpecs = cell(nMjr,1);
      %obj.ftrs = cell(nMjr,1);
      obj.ftrsUse = nan(nMjr,nMnr,MM,nUse);
      
      obj.fernThresh = nan(nMjr,nMnr,MM);
      obj.fernCounts = zeros(nMjr,nMnr,2^MM);
      obj.fernOutput = nan(nMjr,nMnr,2^MM,obj.mdlD);
      obj.fernTS = -inf*ones(nMjr,nMnr);
    end
    
    function p_t = propagate(obj,I,bboxes,p0,pIidx,varargin) % obj const
      % Propagate shapes through regressor cascade.
      %
      % I: [N] Cell array of images
      % bboxes: [Nx2*d]
      % p0: [QxD] initial shapes, absolute coords. M=N*RT1
      % pIidx: [Q] indices into I for rows of p0
      %
      % p_t: [QxDx(T+1)] All shapes over time. p_t(:,:,1)=p0; p_t(:,:,end)
      % is shape after T'th major iteration.
      %
      
      [t0,hWB] = myparse(varargin,...
        't0',1,... % initial/starting major iteration
        'hWaitBar',[]);
      tfWB = ~isempty(hWB);
  
      NI = numel(I);
      assert(isequal(size(bboxes),[NI 2*obj.mdld]));
      [Q,D] = size(p0);
      assert(numel(pIidx)==Q && all(ismember(pIidx,1:NI)));
      assert(D==obj.mdlD);
  
      model = obj.mdl;
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
              
        % TODO: actually only need to compute that subset of features that 
        % is actually used by microregressors
        X = obj.computeFeatures(t,I,bboxes,p,pIidx); 

        % Compute shape correction (normalized units) by summing over
        % microregressors
        pDel = zeros(Q,D);
        for u=1:obj.nMinor
          iFtr = squeeze(obj.ftrsUse(t,u,:,:)); % [MxnUse]
          nUse = size(iFtr,2);
          switch obj.ftrPrm.metatype
            case 'single'
              assert(nUse==1);
              x = X(:,iFtr);
            case 'diff'
              assert(nUse==2);
              x = X(:,iFtr(:,1))-X(:,iFtr(:,2));
          end
          thrs = squeeze(obj.fernThresh(t,u,:));
          inds = fernsInds(x,uint32(1:obj.M),thrs(:)'); 
          yFern = squeeze(obj.fernOutput(t,u,inds,:));
          assert(ndims(yFern)==2); %#ok<ISMAT>
          pDel = pDel + yFern; % normalized units
        end
        
        if obj.regPrm.USE_AL_CORRECTION
          p1 = shapeGt('projectPose',model,p,bbs); % p1 is normalized        
          p = Shape.applyRIDiff(p1,pDel,1,3); % XXXAL HARDCODED HEAD/TAIL
        else
          p = shapeGt('compose',model,pDel,p,bbs); % p (output) is normalized
        end
        p = shapeGt('reprojectPose',model,p,bbs); % back to absolute coords
        p_t(:,:,t+1) = p;
      end
    end
    
    function ftrs = computeFeatures(obj,t,I,bboxes,p,pIidx) % obj const
      % t: major iteration
      % I: [N] Cell array of images
      % bboxes: [Nx2*d]
      % p: [QxD] shapes, absolute coords.
      % pIidx: [Q] indices into I for rows of p
      
      fspec = obj.ftrSpecs{t};
      assert(~isempty(fspec),'No feature specifications for major iteration %d.',t);
      switch fspec.type
        case 'kborig_hack'
          ftrs = shapeGt('ftrsCompKBOrig',obj.mdl,p,I,fspec,...
            pIidx,[],bboxes,obj.regPrm.occlPrm);
        case {'1lm' '2lm' '2lmdiff'}
          ftrs = shapeGt('ftrsCompDup2',obj.mdl,p,I,fspec,...
            pIidx,[],bboxes,obj.regPrm.occlPrm);
        otherwise
          assert(false,'Unrecognized feature specification type.');
      end
    end
    
    function pAll = train(obj,I,bboxes,pGT,p0,pIidx,varargin)
      % 
      %
      % I: [N] cell array of images
      % bboxes: [Nx2*d]
      % pGT: [NxD] GT labels (absolute coords)
      % p0: [QxD] initial shapes (absolute coords)
      % pIidx: [Q] indices into I
      %
      % pAll: [QxDxT+1] propagated training shapes (absolute coords)
      
      [verbose,hWB] = myparse(varargin,...
        'verbose',1,...
        'hWaitBar',[]);
      
      NI = numel(I);
      assert(isequal(size(bboxes),[NI 2*obj.mdld]));
      assert(isequal(size(pGT),[NI obj.mdlD]));
      [Q,D] = size(p0);
      assert(D==obj.mdlD);
      assert(numel(pIidx)==Q);

      T = obj.nMajor;
      pAll = zeros(Q,D,T+1);
      pAll(:,:,1) = p0;
      t0 = 1;
      pCur = p0;
      bboxesFull = bboxes(pIidx,:);
      
      obj.init();
                  
      loss = mean(shapeGt('dist',model,pCur,pGT));
      if verbose
        fprintf('  t=%i/%i       loss=%f     \n',t0-1,T,loss);
      end
      tStart = clock;
      
      prmFtr = obj.ftrPrm;
      ftrRadiusOrig = prmFtr.radius; % for t-dependent ftr radius
      prmReg = obj.regPrm;
      
      for t=t0:T
        if prmReg.USE_AL_CORRECTION
          pCurN_al = shapeGt('projectPose',model,pCur,bboxesFull);
          pGtN_al = shapeGt('projectPose',model,pGT,bboxesFull);
          assert(isequal(size(pCurN_al),size(pGtN_al)));
          pDiffN_al = Shape.rotInvariantDiff(pCurN_al,pGtN_al,1,3); % XXXAL HARDCODED HEAD/TAIL
          pTar = pDiffN_al;
        else
          pTar = shapeGt('inverse',model,pCur,bboxesFull); % pCur: absolute. pTar: normalized
          pTar = shapeGt('compose',model,pTar,pGT,bboxesFull); % pTar: normalized
        end
        
        if numel(ftrRadiusOrig)>1
          prmFtr.radius = ftrRadiusOrig(min(t,numel(ftrRadiusOrig)));
        end
        
        % Generate feature specs; compute features for current training shapes
        switch prmFtr.type
          case {'kborig_hack'}
            fspec = shapeGt('ftrsGenKBOrig',model,prmFtr);
          case {'1lm' '2lm' '2lmdiff'}
            fspec = shapeGt('ftrsGenDup2',model,prmFtr);
        end
        obj.ftrSpecs{t} = fspec;
        X = obj.computeFeatures(t,I,bboxes,pCur,pIidx);
        
        % Regress
        prmReg.ftrPrm = prmFtr;
        [regInfo,pDel] = regTrain(X,pTar,prmReg); 
        assert(iscell(regInfo) && numel(regInfo)==obj.nMinor);
        for u=1:obj.nMinor
          ri = regInfo{u};
          obj.ftrsUse(t,u,:,:) = ri.fids';
          obj.fernThresh(t,u,:) = ri.thrs;
          obj.fernCounts(t,u,:) = nan;
          obj.fernOutput(t,u,:,:) = ri.ysFern;
          obj.fernTS(t,u) = now();          
        end
                  
        % Apply pDel
        if prmReg.USE_AL_CORRECTION
          pCur = Shape.applyRIDiff(pCurN_al,pDel,1,3); %XXXAL HARDCODED HEAD/TAIL
          pCur = shapeGt('reprojectPose',model,pCur,bboxesFull);
        else
          pCur = shapeGt('compose',model,pDel,pCur,bboxesFull);
          pCur = shapeGt('reprojectPose',model,pCur,bboxesFull);
        end
        pAll(:,:,t+1) = pCur;
        
        errPerEx = shapeGt('dist',model,pCur,pGT);
        loss = mean(errPerEx);        
        if verbose
          msg = tStatus(tStart,t,T);
          fprintf(['  t=%i/%i       loss=%f     ' msg],t,T,loss);
        end
        
%         if loss<1e-5
%           T=t;
%           break;
%         end
      end
    end
       
    function update(obj,I,pGT,p0,pIidx)
      
    end
    
  end
    
end
  
  