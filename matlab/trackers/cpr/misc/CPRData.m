classdef CPRData < handle
  
  properties
    Name    % Name of this CPRData
    MD      % [NxR] Table of Metadata
    
    I       % [NxnView] cell vec, images
    pGT     % [NxD] GT shapes for I
    bboxes  % [Nx2d] bboxes for I 
    
    Ipp     % [NxnView] cell vec of preprocessed 'channel' images. .iPP{i,iView} is [nrxncxnchan]
    IppInfo % [nchan] cellstr describing each channel (3rd dim) of .Ipp
    
    H0      % [nbin x nView] for histeq. Currently this is "dumb state" 
    
    iTrn    % [1xNtrn] row indices into I for training set
    iTst    % [1xNtst] row indices into I for test set
  end
  properties (Dependent)
    N
    nView
    d
    D
    nfids
    
    isLabeled % [Nx1] logical, whether trial N has labels    
    isFullyLabeled % [Nx1] logical
    NFullyLabeled

    iUnused % trial indices (1..N) that are in neither iTrn or iTst
    
    NTrn
    NTst
    ITrn % [NTrn x nView]
    ITst % [NTst x nView]
    pGTTrn
    pGTTst
    bboxesTrn
    bboxesTst
    MDTrn
    MDTst
  end
  
  %% Dep prop getters
  methods 
    function v = get.N(obj)
      v = size(obj.I,1);
    end
    function v = get.nView(obj)
      v = size(obj.I,2);
    end
    function v = get.d(obj)
      v = size(obj.bboxes,2)/2;
    end
    function v = get.D(obj)
      v = size(obj.pGT,2);
    end
    function v = get.nfids(obj)
      v = obj.D/obj.d;
    end
    function v = get.isLabeled(obj)
      p = obj.pGT;
      tmp = ~isnan(p);
      tfAllPtsLbled = all(tmp,2); 
      tfAnyPtsLbled = any(tmp,2);
      
      tfPartialLbl = tfAnyPtsLbled & ~tfAllPtsLbled;
      if any(tfPartialLbl)
        n = nnz(tfPartialLbl);
        fprintf(2,'%d trials are partially labeled.\n',n);
      end
      
      v = tfAnyPtsLbled;
    end
    function v = get.isFullyLabeled(obj)
      v = all(~isnan(obj.pGT),2); 
    end
    function v = get.NFullyLabeled(obj)
      v = nnz(obj.isFullyLabeled);
    end
    function v = get.iUnused(obj)
      if ~isempty(intersect(obj.iTrn,obj.iTst))
        warning('CPRData:partition','Overlapping iTrn/iTst.');
      end
      iTrnTst = union(obj.iTrn,obj.iTst);
      v = setdiff(1:obj.N,iTrnTst);
    end
    function v = get.NTrn(obj)
      v = numel(obj.iTrn);
    end
    function v = get.NTst(obj)
      v = numel(obj.iTst);
    end
    function v = get.ITrn(obj)
      v = obj.I(obj.iTrn,:);
    end
    function v = get.ITst(obj)
      v = obj.I(obj.iTst,:);
    end
    function v = get.pGTTrn(obj)
      v = obj.pGT(obj.iTrn,:);
    end
    function v = get.pGTTst(obj)
      v = obj.pGT(obj.iTst,:);      
    end
    function v = get.bboxesTrn(obj)
      v = obj.bboxes(obj.iTrn,:);
    end
    function v = get.bboxesTst(obj)
      v = obj.bboxes(obj.iTst,:);
    end
    function v = get.MDTrn(obj)
      md = obj.MD;
      if isempty(md)
        v = [];
      else          
        v = md(obj.iTrn,:);
      end
    end
    function v = get.MDTst(obj)
      md = obj.MD;
      if isempty(md)
        v = [];
      else
        v = md(obj.iTst,:);
      end
    end
  end
  
  %% 
  methods
    
    function obj = CPRData(varargin)
      % obj = CPRData(movFiles)
      % obj = CPRData(lblFiles,tfAllFrames)
      % obj = CPRData(I,tblP)
      % obj = CPRData(I,tblP,bboxes)
      % obj = CPRData(movFiles,lpos,lpostags,type,varargin)
      % obj = CPRData(movFiles,lpos,lpostags,iMov,frms,varargin)
      
      switch nargin
        case 0
          error('CPRData:CPRData','Invalid number of input arguments.');
        case 1
          movFiles = varargin{1};
          [Is,bb,md] = CPRData.readMovs(movFiles);
          p = nan(size(Is,1),0);
        case 2
          if iscell(varargin{1}) && istable(varargin{2})
            [Is,tblP] = deal(varargin{:});
            p = tblP.p;
            md = tblP;
            md(:,'p') = [];
          else
            assert(false,'Unsupported');
          end          
          if size(Is,2)==1
            sz = cellfun(@(x) [size(x,2),size(x,1)],Is,'uni',0);
            bb = cellfun(@(x)[[1 1] x],sz,'uni',0);
          else
            warningNoTrace('CPRData:bb',...
              'Multiview CPRData. Deferring computation of bounding boxes.');
            bb = nan(size(Is,1),0);
          end
        case 3
          [Is,tblP,bb] = deal(varargin{:});
          p = tblP.p;
          md = tblP;
          md(:,'p') = [];
        otherwise % 4+         
          assert(false,'Unsupported');
      end
      
      assert(iscell(Is));
      N = size(Is,1);
      if iscell(bb)
        bb = cat(1,bb{:});
      end
      assert(isequal(N,size(md,1),size(p,1),size(bb,1)));

      obj.MD = md;
      obj.I = Is;
      obj.pGT = p;
      obj.bboxes = bb;
    end
    
    function append(obj,varargin)
      % Cat/append additional CPRDatas
      % 
      % data.append(data1,data2,...)

      obj.iTrn = obj.iTrn(:)';
      obj.iTst = obj.iTst(:)';
      for i = 1:numel(varargin)
        dd = varargin{i};
        assert(dd.nView==obj.nView || obj.N==0,'Number of views differ for data index %d.',i);
        assert(isequaln(dd.H0,obj.H0),'Different H0 found for data index %d.',i);
        assert(isequal(dd.IppInfo,obj.IppInfo),...
          'Different IppInfo found for data index %d.',i);
        
        Nbefore = size(obj.I,1);
        
        if isempty(obj.MD)
          % Fieldnames of initial obj.MD vs dd.MD might differ eg for 
          % multiTarget; special-case empty obj.MD to deal with this.
          obj.MD = dd.MD;
        else
          obj.MD = cat(1,obj.MD,dd.MD);
        end
        if isempty(obj.I)
          tmpI = [];
        else
          tmpI = obj.I;
        end
        obj.I = cat(1,tmpI,dd.I);
        if isempty(obj.pGT)
          obj.pGT = dd.pGT;
        else
          obj.pGT = cat(1,obj.pGT,dd.pGT);
        end
        obj.bboxes = cat(1,obj.bboxes,dd.bboxes);
        obj.Ipp = cat(1,obj.Ipp,dd.Ipp);
        
        obj.iTrn = cat(2,obj.iTrn,dd.iTrn(:)'+Nbefore);
        obj.iTst = cat(2,obj.iTst,dd.iTst(:)'+Nbefore);
      end
    end
    
    function rmRows(obj,idxrm)
      % remove rows
      %
      % idx: index vector, not logical vec
     
      obj.MD(idxrm,:) = [];
      obj.I(idxrm,:) = [];
      obj.pGT(idxrm,:) = [];
      obj.bboxes(idxrm,:) = [];
      if ~isempty(obj.Ipp)
        obj.Ipp(idxrm,:) = [];
      end
      obj.iTrn = setdiff(obj.iTrn,idxrm);
      obj.iTst = setdiff(obj.iTst,idxrm);    
    end
    
    function tfRm = movieRemap(obj,mIdxOrig2New)
      % mIdxOrig2New: containers.Map, int32 keys and values. 
      %   mIdxOrig2New(oldIdx)==newIdx where oldIdx and/or newIdx can be 
      %   negative.
      
      [obj.MD,tfRm] = MFTable.remapIntegerKey(obj.MD,'mov',mIdxOrig2New);
      obj.I(tfRm,:) = [];
      obj.pGT(tfRm,:) = [];
      obj.bboxes(tfRm,:) = [];
      if ~isempty(obj.Ipp)
        obj.Ipp(tfRm,:) = [];
      end
      obj.iTrn = [];
      obj.iTst = [];
    end
    
    function [tblPnew,tblPupdate] = tblPDiff(obj,tblP) % obj const
      % Compare tblP to .p and .MD wrt MFTable.FLDSCORE

      tbl0 = obj.MD;
      tbl0.p = obj.pGT;
      [tblPnew,tblPupdate] = MFTable.tblPDiff(tbl0,tblP);
    end
  
  end
  
  %% misc data
  methods (Static)
    
    function [I,bb,md] = readMovs(movFiles)
      % movFiles: [N] cellstr
      % Optional PVs:
      % 
      % I: [Nx1] cell array of images (frames)
      % bb: [Nx2d] bboxes
      % md: [Nxm] metadata table
      %
      % Single-view only.
      %
      % No callsites in APT app

      assert(false,'Unsupported');
      
      assert(iscellstr(movFiles));
      nMov = numel(movFiles);

      mr = MovieReader();
      I = cell(0,1);
      sMD = struct('mov',cell(0,1),'frm',[]);
      
      for iMov = 1:nMov
        movName = movFiles{iMov};
        mr.open(movName);
        nf = mr.nframes;
        fprintf('Mov: %s, nframes %d.\n',movName,nf);        
               
        ITmp = cell(nf,1);
        for f = 1:nf
          im = mr.readframe(f);

          if mod(f,10)==0
            fprintf('read frame %d/%d\n',f,nf);
          end
          
          ITmp{f} = im;          
          sMD(end+1,1).mov = movName; %#ok<AGROW>
          sMD(end).frm = f;
        end
        
        I = [I;ITmp]; %#ok<AGROW>
      end
      
      sz = cellfun(@(x)size(x'),I,'uni',0);
      bb = cellfun(@(x)[[1 1] x],sz,'uni',0);

      assert(isequal(numel(sMD),numel(I),size(bb,1)));
      md = struct2table(sMD);
    end
    
    function [I,nmask,didread,tformA] = getFrames(tblMF,varargin)
      % Read frames from movies given MF table
      %
      % tblMF: [NxR] MFTable. tblMF.mov is [NxnView] with nView>1 for
      % multiview data. 
      % 
      % I: [NxnView] cell vector of images for each row of tbl
      % nmask: [NxnView] number of other CCs masked for each im
      % didread: [NxnView] logical
      % tformA: [3x3xNxnView] array. affine tform matrices used when .roi
      %   is present and either ROI-cropping is done or target-cropping via
      %   rotateImsUp==true is done. This is only safe to use
      %   when isDLpipeline is true. These tforms are those that should be
      %   applied to the raw/orig lbls to match the transforms/crops
      %   applied to generate the images in I. tformA(:,:,irow,ivw) is 
      %   the tform mat for irow,ivw.
      
      nView = size(tblMF.mov,2);
      
      [wbObj,forceGrayscale,preload,movieInvert,roiPadVal,...
        rotateImsUp,isDLpipeline,... %rotateImsHeadTail,rotateImsNumPhysPts,...
        doBGsub,...
        bgReadFcn,bgType,trxCache,maskNeighbors,maskNeighborsMeth,empPDF,...
        fgThresh,lObj,usePreProcData] = ...
        myparse(varargin,...
          'wbObj',[],... wbObj: WaitBarWithCancel. If canceled, I will be 'incomplete', ie partially filled.
          'forceGrayscale',true,... 
          'preload',false,...
          'movieInvert',false(1,nView),...
          'roiPadVal',0,... % used when tblMF has .roi
          'rotateImsUp',false,... % if true, rotate all ims so that shape points "up" before cropping.
                              ... % assumes tblMF has .roi and trxCache is specified
                              ... % 'rotateImsHeadTail',[],...  % used when rotateImsUp=true. [ihead itail] landmark/pt indices 
                              ... % 'rotateImsNumPhysPts',nan,... % used when rotateImsUp=true. Number of physical points in tblMF.p. Just a check/assert
          'isDLpipeline',false,... % if true, check/assert things relevant to DL preproc pipe.
          'doBGsub',false,... % if true, I will contain bg-subbed images
          'bgReadFcn',[],... % [bg,bgdev] = fcn(movieFile,movInfo)
                         ... % reads/generates bg image for given movie
          'bgType','other',... % one of {'light on dark','dark on light','other'}
          'trxCache',[],...
          'maskNeighbors',0,... % if true, neighbor-masking is performed
          ...   % BEGIN USED when maskNeighbors==true;
          'maskNeighborsMeth','Conn. Comp',...
          'maskNeighborsEmpPDF',[],... %used if maskNeighborsMeth=='Emp. PDF'
          'fgThresh',nan,...   % END USED for maskNeighbors
          'labeler',[],...
          'usePreProcData',false...
          );
      assert(numel(movieInvert)==nView);
      if usePreProcData,
        assert(~isempty(lObj) && ~rotateImsUp);
      end
        
      if doBGsub && roiPadVal~=0
        warningNoTrace('Background subtraction enabled. Setting roi pad value to 0.');
%         roiPadVal = 0;
      end
      tfWB = ~isempty(wbObj);
      
      N = height(tblMF);      
  
      tfROI = tblfldscontains(tblMF,'roi');
      if tfROI
        roi = tblMF.roi;
      end
      
      if rotateImsUp
        assert(tfROI && ~isempty(trxCache));
        %assert(numel(rotateImsHeadTail)==2);
        tblfldscontainsassert(tblMF,'p');
        %assert(size(tblMF.p,2)==nView*rotateImsNumPhysPts*2);
      end

      if isDLpipeline
        assert(~doBGsub);
        assert(~maskNeighbors);
      end
      
      % Initialize outputs early, we may early return if user cancels wb.
      I = cell(N,nView);
      didread = false(N,nView);
      nmask = zeros(N,nView);
      tformA = nan(3,3,N,nView);
            
      for iVw=1:nView
        [movsUn,~,movidx] = unique(tblMF.mov(:,iVw));
        
        if tfWB
          wbObj.startPeriod(sprintf('Reading images: view %d',iVw),...
            'shownumden',true,'denominator',N);          
        end
        
        nframesAttemptRead = 0;
        for iMov=1:numel(movsUn)
          if tfWB
            tfCancel = wbObj.updateFracWithNumDen(nframesAttemptRead);
            if tfCancel
              wbObj.endPeriod();
              return;
            end
          else
            fprintf('Movie %s (%d/%d)\n',movsUn{iMov},iMov,numel(movsUn));
          end
          
          mov = movsUn{iMov};
          
          if usePreProcData,
            
          else
            mr = MovieReader();
            mr.forceGrayscale = forceGrayscale;
            mr.flipVert = movieInvert(iVw);
            mr.preload = preload;
            %mr.neednframes = false;
            mr.open(mov,'bgType',bgType,'bgReadFcn',bgReadFcn);
          end
          
          % Note: we don't setCropInfo here; cropping handled explicitly
          % b/c most of the time it comes from the trx
          
          idxcurr = find(movidx == iMov);
          nframesAttemptRead = nframesAttemptRead + numel(idxcurr);
          for iiTrl = 1:numel(idxcurr)
            iTrl = idxcurr(iiTrl);
                        
            trow = tblMF(iTrl,:);
            f = trow.frm;
            iTgt = trow.iTgt;
            assert(strcmp(mov,trow.mov{iVw}));
            
            if tfROI
              % Will be handy below
              roiVw = roi(iTrl,(1:4)+4*(iVw-1)); % [xlo xhi ylo yhi]
              roiXlo = roiVw(1);
              roiXhi = roiVw(2);
              roiYlo = roiVw(3);
              roiYhi = roiVw(4);
            end
            
            if usePreProcData,
              I(iTrl,iVw) = lObj.preProcData.ITrn(iTrl,iVw);
              didread(iTrl,iVw) = true;
              % manually make a translation tform consistent with this
              % padgrab. Could just use CropImAroundTrx all the time.
              if tfROI
                Ttmp = [1 0 0;0 1 0;-(roiXlo-1) -(roiYlo-1) 1];
                tformA(:,:,iTrl,iVw) = Ttmp;
              else
                tformA(:,:,iTrl,iVw) = eye(3);
              end
              continue;
            end
            
            try
            
            if maskNeighbors
              assert(~isDLpipeline);
              assert(~rotateImsUp,'Currently unsupported.');
              
              [imraw,imOrigTy] = mr.readframe(f,'doBGsub',false);
              imdiff = PxAssign.simplebgsub(mr.bgType,double(imraw), ...
                mr.bgIm,mr.bgDevIm); % Note: mr.flipVert is NOT applied to .bgIm, .bgDevIm
              % imdiff has scale per ~imOrigTy
              
              tfile = trow.trxFile{iVw};
              trx = Labeler.getTrxCacheStc(trxCache,tfile,mr.nframes);
              
              % Currently we mask the entire image even if we only care about
              % a zoomed-in roi
              switch maskNeighborsMeth
                case 'Conn. Comp'
                  imL = PxAssign.asgnCCcore(imdiff,trx,f,fgThresh);
                case 'GMM-EM'
                  imL = PxAssign.asgnGMMglobalcore(imdiff,trx,f,fgThresh);
                case 'Emp. PDF'
                  if isempty(empPDF)
                    error('No empirical PDF has been generated/stored for this project. Call the ''updateFGEmpiricalPDF'' Labeler method first.');
                  end
                  if ~isequal(empPDF.prmBackSub.BGType,bgType)
                    warningNoTrace('Stored empirical PDF has background type (%s) that differs from current background type (%s).',...
                      empPDF.prmNborMask.BGType,bgType);
                  end
                  if ~isequal(empPDF.prmNborMask.FGThresh,fgThresh)
                    warningNoTrace('Stored empirical PDF has foreground threshold (%.2f) that differs from current foreground threshold (%.2f).',...
                      empPDF.prmNborMask.FGThresh,fgThresh);
                  end
                  imL = PxAssign.asgnPDF(imdiff,trx,f,...
                    empPDF.fgpdf,empPDF.xpdfctr,empPDF.ypdfctr,...
                    empPDF.amu,empPDF.bmu,'fgthresh',fgThresh);
                otherwise
                  assert(false,'Unrecognized neighbor-masking method.');
              end
              
              if doBGsub
                % bgsub ON, nbor masking ON.
                % We will be masking imdiff with zeros. imdiff is a double
                % with original scale/range
                imToMask = imdiff;
                imBGToApply = zeros(size(imdiff));
                % roiPadVal should be 0 here since doBGsub is on
              else
                % bgsub OFF, nbor masking ON.
                % We will be masking imraw with movieReader.bgIm. imraw could
                % have arbitrary type here, but .bgIm is expected to have the
                % same scale.
                imToMask = double(imraw);
                imBGToApply = mr.bgIm;
              end
              
              if tfROI
                IMBGPADVAL = nan; % irrelevant, no effect as masking should not occur outside image
                imToMaskRoi = padgrab(imToMask,roiPadVal,roiYlo,roiYhi,roiXlo,roiXhi);
                imBGToApplyRoi = padgrab(imBGToApply,IMBGPADVAL,roiYlo,roiYhi,roiXlo,roiXhi);
                imLroi = padgrab(imL,0,roiYlo,roiYhi,roiXlo,roiXhi);
                [nmask(iTrl,iVw),imroi] = PxAssign.performMask(...
                  imToMaskRoi,imBGToApplyRoi,imLroi,trx,iTgt,f,'imroi',roiVw);
              else
                [nmask(iTrl,iVw),imroi] = PxAssign.performMask(...
                  imToMask,imBGToApply,imL,trx,iTgt,f);
              end
              
              % Rescale to [0,1] for 'usual' types
              imroi = PxAssign.imRescalePerType(imroi,imOrigTy);
              
              % As in other branch, imroi could have varying type here.
            else
              [im,imOrigTy] = mr.readframe(f,'doBGsub',doBGsub);
              
              if doBGsub
                % BGsub leaves im as a double but scaled as in original
                im = PxAssign.imRescalePerType(im,imOrigTy);
              end
              
              if tfROI
                if rotateImsUp
                  tfile = trow.trxFile{iVw};
                  trx = Labeler.getTrxCacheStc(trxCache,tfile,mr.nframes);
                  [trxx,trxy,trxth] = readtrx(trx,f,iTgt); % using headtailth instead of trxth
    %                   prow = trow.p; % doesn't matter if .p is relative to roi or absolute
    %                   prow = reshape(prow,[rotateImsNumPhysPts nView 2]);
    %                   htxy = squeeze(prow(rotateImsHeadTail,iVw,:)); % {head/tail},{x/y}
    %                   htdxy = htxy(1,:)-htxy(2,:); % head-tail
    %                   htth = atan2(htdxy(2),htdxy(1));                  
                  roiDX = roiXhi-roiXlo; % span is <this>+1, expected to be odd
                  roiDY = roiYhi-roiYlo;
                  assert(roiDX==roiDY && mod(roiDX+1,2)==0,...
                    'Expected square roi crop centered around trx.');
                  roiRad = roiDX/2; 
                  [imroi,Atmp] = CropImAroundTrx(...
                    im,trxx,trxy,trxth,roiRad,roiRad,'fillvalues',roiPadVal);
                  % Atmp transforms so that the trx center is located at
                  % 0,0. We want it to be at (roiRad+1,roiRad+1).
                  Atmp(end,[1 2]) = Atmp(end,[1 2]) + roiRad + 1;
                  tformA(:,:,iTrl,iVw) = Atmp;
                else
                  if ndims(im) == 2 %#ok<ISMAT>
                    imroi = padgrab(im,roiPadVal,roiYlo,roiYhi,roiXlo,roiXhi);
                  elseif ndims(im) == 3
                    imroi = padgrab(im,roiPadVal,roiYlo,roiYhi,roiXlo,roiXhi,1,3);
                  else
                    error('Undefined number of channels');
                  end
                  % manually make a translation tform consistent with this
                  % padgrab. Could just use CropImAroundTrx all the time.
                  Ttmp = [1 0 0;0 1 0;-(roiXlo-1) -(roiYlo-1) 1];
                  tformA(:,:,iTrl,iVw) = Ttmp;
                end
              else
                imroi = im;
                tformA(:,:,iTrl,iVw) = eye(3);
              end
              
              % At this point, im could have varying type depending on movie
              % format, doBGsub, etc. See MovieReader/readframe.
            end
            
            I{iTrl,iVw} = imroi;
            didread(iTrl,iVw) = true;
            
            catch ME,
              warning('Could not read frame %d from %s:\n%s\n',f,mov,getReport(ME));
            end
            
          end
          if exist('mr','var') && mr.isOpen
            mr.close();
          end
        end
        if tfWB
          wbObj.endPeriod();
        end
      end
    end

    function bboxes = getBboxes2D(I)
      % Compute default 2D bounding boxes for image set
      %
      % I: [N] column cell vec of images
      % 
      % bboxes: [Nx4] 2d bboxes
      
      if iscell(I),
        assert(iscolumn(I));
        sz = cellfun(@(x)[size(x,2) size(x,1)],I,'uni',0);
        bboxes = cellfun(@(x)[[1 1] x],sz,'uni',0);
        bboxes = cat(1,bboxes{:});
      else
        N = size(I.imoffs,1);
        bboxes = [ones(N,2),I.imszs([2 1],:)'];
      end
    end

  end
  
  %% PreProc
  methods
    
    function vizHist(obj,varargin)
      [g,smoothspan,nbin] = myparse(varargin,...
        'g',[],... [N] grouping vector, numeric or categorical.
        'smoothspan',nan,...
        'nbin',256);
      tfsmooth = ~isnan(smoothspan);
      
      assert(obj.nView==1,'Single-view only.');
      
      H = nan(nbin,obj.N);
      for i = 1:obj.N
        H(:,i) = imhist(obj.I{i},nbin);
      end
      
      if isempty(g)
        g = ones(obj.N,1);
      end
      assert(isvector(g) && numel(g)==obj.N);
      
      gUn = unique(g);
      nGrp = numel(gUn);
      muGrp = nan(nbin,nGrp);
      sdGrp = nan(nbin,nGrp);
      for iGrp = 1:nGrp
        gCur = gUn(iGrp);
        tf = g==gCur;
        Hgrp = H(:,tf);
        fprintf('Working on grp=%d, n=%d.\n',iGrp,nnz(tf));
        muGrp(:,iGrp) = nanmean(Hgrp,2);
        sdGrp(:,iGrp) = nanstd(Hgrp,[],2);
        
        if tfsmooth
          muGrp(:,iGrp) = smooth(muGrp(:,iGrp),smoothspan);
          sdGrp(:,iGrp) = smooth(sdGrp(:,iGrp),smoothspan);
        end
      end
      
      figure;
      x = 1:nbin;
      plot(x,muGrp,'linewidth',2);
      legend(arrayfun(@num2str,1:nGrp,'uni',0));
      hold on;
      ax = gca;
      ax.ColorOrderIndex = 1;
      plot(x,muGrp-sdGrp);
      ax.ColorOrderIndex = 1;
      plot(x,muGrp+sdGrp);
      
      grid on;
    end
    
    function H0 = histEq(~,varargin)
      % Perform histogram equalization on all images
      %
      % Optional PVs:
      % H0: [nbin] intensity histogram used in equalization
      % g: [N] grouping vector, either numeric or categorical. Images with
      % the same value of g are histogram-equalized together. For example,
      % g might indicate which movie the image is taken from.
      
      H0 = [] ;
      assert(false,'Deprecated codepath.');
      
%       [H0,nbin,g,wbObj] = myparse(varargin,...
%         'H0',[],...
%         'nbin',256,...
%         'g',ones(obj.N,1),...
%         'wbObj',[]); % WaitBarWithCancel. If canceled, obj UNCHANGED and H0 indeterminate
%       tfH0Given = ~isempty(H0);
%       
%       assert(obj.nView==1,'Single-view only.');
%       assert(numel(g)==obj.N);
%       
%       imSz = cellfun(@size,obj.I,'uni',0);
%       cellfun(@(x)assert(isequal(x,imSz{1})),imSz);
%       imSz = imSz{1};
%       
%       if ~tfH0Given
%         H0 = typicalImHist(obj.I,'nbin',nbin,'hWB',wbObj);
%         if wbObj.isCancel
%           return;
%         end
%       end
%       obj.H0 = H0;
%       
%       % normalize one group at a time
%       gUn = unique(g);
%       nGrp = numel(gUn);
%       fprintf(1,'%d groups to equalize.\n',nGrp);
%       for iGrp = 1:nGrp
%         gCur = gUn(iGrp);
%         tf = g==gCur;
%         
%         bigim = cat(1,obj.I{tf});
%         bigimnorm = histeq(bigim,H0);
%         obj.I(tf) = mat2cell(bigimnorm,...
%           repmat(imSz(1),[nnz(tf) 1]),imSz(2));
%       end
    end

    function computeIpp(obj,sig1,sig2,iChan,varargin)
      % Preprocess images and set .Ipp.
      %
      % sig1,sig2: see Features.pp
      % iChan: index vectors into channels for which channels to keep/store.
      %
      % Optional PVs:
      % - iTrl. trial indices for which Ipp should be computed. Defaults to
      % find(obj.isFullyLabeled).
      % - jan. Use jan values for sig1,sig2,iChan.
      % - romain. Use romain values for sig1,sig2,iChan.
      % - See Features.pp for other optional pvs
      
      [iTrl,jan,romain,hWB] = myparse(varargin,...
        'iTrl',find(obj.isFullyLabeled),...
        'jan',false,...
        'romain',false,...
        'hWaitBar',[]);
      nTrl = numel(iTrl);
      
      tfBPP = false;
      nVw = obj.nView;
      if jan
        assert(nVw==1);
        fprintf(1,'Using "Jan" settings.\n');
        pause(2);
        sig1 = [0 2 4 8]; % for now, normalizing/rescaling channels assuming these sigs
        sig2 = [0 2 4 8];
        iChan = [...
          2 3 ... % blur sig1(1:3)
          5 6 7 9 10 11 13 14 17 ... % SGS
          22 23 25 26 27 29 30]; % SLS
      elseif iscell(romain) && isa(romain{1},'CPRBlurPreProc')
        bppCell = romain;
        assert(numel(bppCell)==nVw);
        fprintf(1,'Using Romain/CPRBlurPreProc settings.\n');
        for i=1:numel(bppCell)
          fprintf(1,'... %d: %s\n',i,bppCell{i}.name);
        end
        tfBPP = true;
      end
        
      ipp = cell(nTrl,nVw);
      ippInfo = cell(1,nVw);
      for iVw=1:nVw
        if tfBPP
          bpp = bppCell{iVw};
          [S,SGS,SLS] = Features.pp(obj.I(iTrl,iVw),bpp.sig1,bpp.sig2,...
            'sRescale',bpp.sRescale,'sRescaleFacs',bpp.sRescaleFacs,...
            'sgsRescale',bpp.sgsRescale,'sgsRescaleFacs',bpp.sgsRescaleFacs,...
            'slsRescale',bpp.slsRescale,'slsRescaleFacs',bpp.slsRescaleFacs,...
            'sgsRescaleClipPctThresh',2,...
            'hWaitBar',hWB);
          sig1 = bpp.sig1;
          sig2 = bpp.sig2;
          iChan = bpp.iChan;
        else
          [S,SGS,SLS] = Features.pp(obj.I(iTrl,iVw),sig1,sig2,'hWaitBar',hWB);
        end
        n1 = numel(sig1);
        n2 = numel(sig2);
        nPPfull = n1+n1*n2+n1*n2;
        assert(iscell(S) && isequal(size(S),[nTrl n1]));
        assert(iscell(SGS) && isequal(size(SGS),[nTrl n1 n2]));
        assert(iscell(SLS) && isequal(size(SLS),[nTrl n1 n2]));

        for i=1:nTrl
          ipp{i,iVw} = cat(3,S{i,:},SGS{i,:},SLS{i,:});
          assert(size(ipp{i,iVw},3)==nPPfull);
        end
        
        info = arrayfun(@(x)sprintf('S:sig1=%.2f',x),sig1(:),'uni',0);
        for i2 = 1:n2 % note raster order corresponding to order of ipp{iTrl}
          for i1 = 1:n1
            info{end+1,1} = sprintf('SGS:sig1=%.2f,sig2=%.2f',sig1(i1),sig2(i2)); %#ok<AGROW>
          end
        end
        for i2 = 1:n2 % etc
          for i1 = 1:n1
            info{end+1,1} = sprintf('SLS:sig1=%.2f,sig2=%.2f',sig1(i1),sig2(i2)); %#ok<AGROW>
          end
        end
        szassert(info,[nPPfull 1]);
        
        ipp(:,iVw) = cellfun(@(x)x(:,:,iChan),ipp(:,iVw),'uni',0);
        ippInfo{iVw} = info(iChan);
      end
      
      obj.Ipp = cell(obj.N,nVw);
      obj.Ipp(iTrl,:) = ipp;
      if nVw==1
        obj.IppInfo = ippInfo{1};
      else
        obj.IppInfo = ippInfo;
      end
    end
    
    function cnts = channelDiagnostics(obj,iTrl,cnts)
      % Compute diagnostics on .I, .Ipp based in trials iTrl
      
      tfCntsSupplied = exist('cnts','var')>0;
      
      assert(obj.nView==1,'Single-view only.');

      edges = 0:1:256;
      
      if ~tfCntsSupplied
        nedge = numel(edges);
        npp = numel(obj.IppInfo);
        cnts = zeros(nedge,npp+1);
        for iT = iTrl(:)'
          x = obj.I{iT};
          cnts(:,1) = cnts(:,1) + histc(x(:),edges);
          
          for ipp = 1:npp
            x = obj.Ipp{iT}(:,:,ipp);
            cnts(:,ipp+1) = cnts(:,ipp+1) + histc(x(:),edges);
          end
          
          fprintf(1,'Done with iTrl=%d\n',iT);
        end
      end
      
      info = [{'I'}; obj.IppInfo];
      nplot = numel(info);
      assert(nplot==size(cnts,2));
      figure('windowstyle','docked');
      axs = createsubplots(4,ceil(nplot/4));
      for iPlot = 1:nplot
        ax = axs(iPlot);
        axes(ax); %#ok<LAXES>
        
        y = cnts(:,iPlot); % frequency count
        assert(y(end)==0);
        y = y(1:end-1)';
        x = (edges(1:end-1)+edges(2:end))/2; % value
        
        [mu,~,sd,med,mad] = freqCountStats(x,y);
        
        plot(ax,x,log10(y));
        xlim(ax,[0 256]);
        hold(ax,'on');
        yl = ylim(ax);
        plot(ax,[mu mu],yl,'r');
        plot(ax,[med med],yl,'m');
        grid on;
        tstr = sprintf('%s: mad=%.3f, sd=%.3f',info{iPlot},mad,sd);
        title(ax,tstr,'interpreter','none','fontsize',8);
        
        if iPlot~=1
          set(ax,'XTickLabel',[],'YTickLabel',[]);
        end
      end
      linkaxes(axs);
    end
    
    function [Is,nChan] = getCombinedIs(obj,iTrl) % obj CONST
      % Get .I combined with .Ipp for specified trials.
      %
      % iTrl: [nTrl] vector of trials
      %
      % Is: [nTrlxnView] cell vec of image stacks [nr nc nChan] where
      %   nChan=1+numel(obj.Ipp)
      % nChan: number of TOTAL channels used/found
      
      if obj.nView==1
        nChanPP = numel(obj.IppInfo);
      else
        if isempty(obj.IppInfo)
          nChanPP = 0;
        else
          nChanPP = cellfun(@numel,obj.IppInfo);
          nChanPP = unique(nChanPP);
        end
        assert(isscalar(nChanPP));
      end
      fprintf(1,'Using %d additional channels.\n',nChanPP);
      
      nTrl = numel(iTrl);
      nVw = obj.nView;
      Is = cell(nTrl,nVw);
      for i=1:nTrl
        iT = iTrl(i);
        for iVw=1:nVw        
          im = obj.I{iT,iVw};
          if nChanPP==0
            impp = nan(size(im,1),size(im,2),0);
          else
            impp = obj.Ipp{iT,iVw};
          end
          assert(size(impp,3)==nChanPP);
          Is{i,iVw} = cat(3,im,impp);
        end
      end
      
      nChan = nChanPP+1;
    end
    
    function [Iinfo,nChan] = getCombinedIsMat(obj,iTrl) % obj CONST
      % Get .I combined with .Ipp for specified trials.
      %
      % iTrl: [nTrl] vector of trials
      %
      % Iinfo is a struct with the following fields:
      %   Is: vector of all nTrl x nView images strung out in order of rows, pixels, channels, image, view
      %   imszs: [2 x nTrl x nView] size of each image
      %   imoffs: [nTrl x nView] offset for indexing image (i,view) (image will
      %     be from off(i,view)+1:off(i,view)+imszs(1,i,view)*imszs(2,i,view)
      % nChan: number of TOTAL channels used/found
            
      if obj.nView==1
        nChanPP = numel(obj.IppInfo);
      else
        if isempty(obj.IppInfo)
          nChanPP = 0;
        else
          nChanPP = cellfun(@numel,obj.IppInfo);
          nChanPP = unique(nChanPP);
        end
        assert(isscalar(nChanPP));
      end
      fprintf(1,'Using %d additional channels.\n',nChanPP);
      
      Iinfo = struct;
      nTrl = numel(iTrl);
      nVw = obj.nView;
      Iinfo.Is = [];
      Iinfo.imszs = nan([2,nTrl,nVw]);
      Iinfo.imoffs = nan([nTrl,nVw]);
      Iinfo.imoffs(1) = 0;
      for i=1:nTrl
        iT = iTrl(i);
        for iVw=1:nVw        
          im = obj.I{iT,iVw};
          if isa(im,'uint8')
            im = double(im)/255;
          elseif isa(im,'uint16')
            im = double(im)/(2^16-1);
          end
          if nChanPP==0
            impp = nan(size(im,1),size(im,2),0);
          else
            impp = obj.Ipp{iT,iVw};
          end
          assert(size(impp,3)==nChanPP);
          im = cat(3,im,impp);
          szcurr = numel(im);
          offnext = Iinfo.imoffs(i,iVw)+szcurr;
          Iinfo.Is(Iinfo.imoffs(i,iVw)+1:offnext) = im;
          Iinfo.imszs(:,i,iVw) = [size(im,1),size(im,2)];
          if iVw < nVw
            Iinfo.imoffs(i,iVw+1) = offnext;
          elseif i<nTrl
            Iinfo.imoffs(i+1,1) = offnext;
          else
            % last entry; none
          end
          %Iinfo.imoffs(nTrl*(iVw-1)+i+1) = offnext;
        end
      end
      
      nChan = nChanPP+1;
      Iinfo.nChan = nChan;
    end
    
    function [sgscnts,slscnts,sgsedge,slsedge] = calibIppJan(obj,nsamp)
      % Sample SGS/SLS intensity histograms
      %
      % nsamp: number of trials to sample
      
      assert(obj.nView==1);
      
      sig1 = [0 2 4 8]; % for now, normalizing/rescaling channels assuming these sigs
      sig2 = [0 2 4 8];
      n1 = numel(sig1);
      n2 = numel(sig2);
      
      sgsedge = [0:160 inf];
      nbinSGS = numel(sgsedge)-1;
      sgscnts = repmat({zeros(1,nbinSGS)},n1,n2);
      slsedge = [-inf -160:160 inf];
      nbinSLS = numel(slsedge)-1;
      slscnts = repmat({zeros(1,nbinSLS)},n1,n2);
      
      iTrlSamp = randsample(obj.N,nsamp);
      nTrlSamp = numel(iTrlSamp);
      for i = 1:nTrlSamp
        if mod(i,10)==0
          disp(i);
        end
        iTrl = iTrlSamp(i);
        
        [~,SGS,SLS] = Features.pp(obj.I(iTrl),sig1,sig2,'sgsRescale',false,'slsRescale',false);
        for iSGS = 1:numel(SGS)
          sgscnts{iSGS} = sgscnts{iSGS} + histcounts(SGS{iSGS},sgsedge);
        end
        for iSLS = 1:numel(SLS)
          slscnts{iSLS} = slscnts{iSLS} + histcounts(SLS{iSLS},slsedge);
        end
      end
    end
    
  end
  
  methods (Static)
    function [sgsthresh,slsspan] = calibIppJan2(sgscnts,slscnts,sgsedge,slsedge)
      
      assert(obj.nView==1);

      ntmp1 = cellfun(@sum,sgscnts);
      ntmp2 = cellfun(@sum,slscnts);
      n = unique([ntmp1(:);ntmp2(:)]);
      assert(isscalar(n));
      
      sgsCtr = (sgsedge(1:end-1)+sgsedge(2:end))/2;
      slsCtr = (slsedge(1:end-1)+slsedge(2:end))/2;
      
      sgscum = cellfun(@(x)cumsum(x)/n,sgscnts,'uni',0);
      slscum = cellfun(@(x)cumsum(x)/n,slscnts,'uni',0);
      
      SGSTHRESH = .999;
      SLSTHRESH = [.01 .99];
      assert(isequal(size(sgscnts),size(slscnts)));
      [n1,n2] = size(sgscnts);
      sgsthresh = nan(n1,n2);
      slsspan = nan(n1,n2);
      for i1 = 1:n1
        for i2 = 1:n2
          y = sgscum{i1,i2};
          iThresh = find(y>SGSTHRESH,1);
          thresh = sgsCtr(iThresh);
          sgsthresh(i1,i2) = thresh;
          cla;
          plot(sgsCtr,y);
          hold on;
          plot([thresh thresh],[0 1],'r');
          grid on;
          title(sprintf('sgs(%d,%d): thresh ptile %.3f: %.3f\n',i1,i2,SGSTHRESH,thresh),...
            'fontweight','bold','interpreter','none');
          input('hk');
        end
      end
      for i1 = 1:n1
        for i2 = 1:n2
          y = slscum{i1,i2};
          iThresh0 = find(y<SLSTHRESH(1),1,'last');
          iThresh1 = find(y>SLSTHRESH(2),1,'first');
          thresh0 = slsCtr(iThresh0);
          thresh1 = slsCtr(iThresh1);
          slsspan(i1,i2) = thresh1-thresh0;
          cla;
          plot(slsCtr,y);
          hold on;
          plot([thresh0 thresh0],[0 1],'r');
          plot([thresh1 thresh1],[0 1],'r');
          grid on;
          title(sprintf('sls(%d,%d): span %.3f. [%.3f %.3f]\n',i1,i2,...
            slsspan(i1,i2),thresh0,thresh1),'fontweight','bold','interpreter','none');
          input('hk');
        end
      end
    end
  end
  
  %%   
  methods
    
    function varargout = viz(obj,varargin)
      [varargout{1:nargout}] = Shape.viz(obj.I(obj.isFullyLabeled,:),obj.pGT(obj.isFullyLabeled,:),...
        struct('nfids',obj.nfids,'D',obj.D),'md',obj.MD(obj.isFullyLabeled,:),varargin{:});
    end
    function varargout = vizIdx(obj,iTrls,varargin)
      n = numel(iTrls);
      nr = floor(sqrt(n));
      nc = ceil(n/nr);
      [varargout{1:nargout}] = Shape.viz(obj.I,obj.pGT,...
        struct('nfids',obj.nfids,'D',obj.D),'md',obj.MD,...
        'nr',nr,'nc',nc,'idxs',iTrls,varargin{:});
    end
    
    function n = getFilename(obj)
      n = sprintf('td_%s_%s.mat',obj.Name,datestr(now,'yyyymmdd'));
    end
        
    function summarize(obj,gMDFld,iTrl)
      % gMDFld: grouping field in metadata
      % iTrl: vector of trial indices
      
      tMD = obj.MD(iTrl,:);
      tfLbled = obj.isFullyLabeled(iTrl,:);
      g = categorical(tMD.(gMDFld));
      gUn = unique(g);
      nGrp = numel(gUn);
      for iGrp = 1:nGrp
        gCur = gUn(iGrp);
        tfG = g==gCur;
        tfGAndLbled = tfG & tfLbled;
        fprintf(1,'Group (%s): %s. nfrm=%d, nfrmlbled=%d.\n',...
          gMDFld,char(gCur),...
          nnz(tfG),nnz(tfGAndLbled));
      end
    end
    
    function [iSim,sim] = findSimilarFrames(obj,iTrl,iTest)
      % Find frames similar to iTrl
      %
      % iTrl: scalar trial index
      % iTest: vector of trial indices to consider
      %
      % iSim: Same as iTest, but permuted in order of decreasing
      %  similarity
      % sim: Similarity scores corresponding to iSim (will be monotonically
      %  decreasing). Right now this is a regular correlation coef.
      
      assert(obj.nView==1);

      nTest = numel(iTest);
      im0col = double(obj.I{iTrl}(:));
      sim = nan(nTest,1);
      for i = 1:nTest
        im1col = double(obj.I{iTest(i)}(:));
        tmp = corrcoef(im0col,im1col);
        sim(i) = tmp(1,2);
        
        if mod(i,100)==0
          fprintf('%d/%d\n',i,nTest);
        end
      end
      
      [sim,idx] = sort(sim,'descend');
      iSim = iTest(idx);
    end

  end
  
  %% partitions
  methods (Static)
    
    %#%MV
    function [grps,ffd,ffdiTrl] = ffTrnSet(tblP,gvar)
      % Furthest-first training set analysis
      %
      % tblP: table with labeled positions (p)
      % gvar: field to use as grouping var. If empty, all rows in a single
      % group.
      %
      % grps: [Ngrp] categorical, unique groups found
      % ffd: [Ngrp] cell vec. ffd{i} contains a vector of "furthest-first"
      % distances, sorted in decreasing order.
      % ffdiTrl. [Ngrp] cell vec. ffdiTrl{i} is a vector of indices into 
      % tblP for ffd{i}.
            
      pTrn = tblP.p;
      if isempty(gvar)
        g = ones(size(tblP,1),1);
      else
        g = tblP.(gvar);
      end
      g = categorical(g);
      grps = unique(g);
      nGrps = numel(grps);
      ffd = cell(nGrps,1);
      ffdiTrl = cell(nGrps,1);
      
      for iGrp = 1:nGrps
        gCur = grps(iGrp);
        tf = g==gCur;
        iG = find(tf);
        pG = pTrn(iG,:);
        nG = numel(iG);
        
        % use furthestfirst to order shapes by decreasing distance
        warnst = warning('off','backtrace');
        [~,~,tmpidx,~,mindists] = furthestfirst(pG,nG,'Start',[],'hWaitBar',true);  
        warning(warnst);
        
        mindists(1) = inf;
        assert(isequal(sort(mindists,'descend'),mindists));
        
        ffd{iGrp} = mindists;
        ffdiTrl{iGrp} = iG(tmpidx);
      end
    end
    
    %#%MV
    function hFig1 = ffTrnSetSelect(tblP,grps,ffd,ffdiTrl,varargin)
      % Display furthestfirst distances for groups in subplots; enable
      % clicking on subplots to visualize training shape
      
      [fontsz,cbkFcn] = myparse(varargin,...
        'fontsize',8,...
        'cbkFcn',[]); % called when user clicks; signature: cbk(xSel,ySel)
      
      assert(isequal(numel(grps),numel(ffd),numel(ffdiTrl)));
      cellfun(@(x,y)assert(isequal(size(x),size(y))),ffd,ffdiTrl);
      assert(iscategorical(grps));
      
      nGrp = numel(grps);
      nrc = ceil(sqrt(nGrp));
      hFig1 = figure;
      axs = createsubplots(nrc,nrc,.06);
      bdfCbks = cell(nGrp,1);
      for iGrp = 1:nGrp
        gstr = char(grps(iGrp));
        gstr = gstr(1:min(6,end));
        ax = axs(iGrp);
        plot(ax,ffd{iGrp});
        grid(ax,'on');
        title(ax,gstr,'interpreter','none','fontsize',fontsz);
        if iGrp==1
          ylabel(ax,'distance (px^2)','fontsize',fontsz);
        end
        if isempty(cbkFcn)
          cbkFcn = @(x,y)nst(x,y);
        end
        bdfCbks{iGrp} = cbkFcn;
        ax.YScale = 'log';
      end
      
      LiveDataCursor(hFig1,axs,bdfCbks);
      
      function nst(xsel,ysel) %#ok<INUSL>
        % xsel, ysel: (x,y) on ffd plot nearest to user click
        
        iTrnAcc = [];
        for zGrp = 1:nGrp
          ffdists = ffd{zGrp};
          ffidxs = ffdiTrl{zGrp};          
          nTot = numel(ffdists);
          tfSel = ffdists>=ysel;
          nSel = nnz(tfSel);
          fprintf(1,'%s: nSel/nTot=%d/%d (%d%%)\n',char(grps(zGrp)),...
            nSel,nTot,round(nSel/nTot*100));
          iTrnAcc = [iTrnAcc; ffidxs(tfSel)]; %#ok<AGROW>
        end
        nP = size(tblP,1);
        nTrnAcc = numel(iTrnAcc);
        fprintf(1,'Grand total of %d/%d (%d%%) shapes selected for training.\n',...
          nTrnAcc,nP,round(nTrnAcc/nP*100));  
      end
    end

  end
  
  methods 
    
    function ptnHalfHalf(obj)
      % - all lblFile+Mov equally weighted 
      % - for each mov, first half training second half test
      
      tMD = obj.MD;
      lblFile = tMD.lblFile;
      iMov = tMD.iMov;
      expID = strcat(lblFile,'#',num2str(iMov));
      assert(numel(expID)==obj.N);
      
      expIDUn = unique(expID);
      expIDUnCnt = cellfun(@(x)nnz(strcmp(expID,x)),expIDUn);
      nExpUn = numel(expIDUn);
      nFrmLCD = min(expIDUnCnt); % "lowest common denominator"
      nTrnTst = floor(nFrmLCD/2);
      fprintf('%d exps (lblfile+mov). nFrmLCD=%d. nTrn or nTst=%d\n',nExpUn,nFrmLCD,nTrnTst);      
      
      iTrnAcc = zeros(1,0);
      iTstAcc = zeros(1,0);
      for iExp = 1:nExpUn
        eID = expIDUn{iExp};
        tf = strcmp(expID,eID);
        assert(nnz(tf)==expIDUnCnt(iExp));        

        % first nFrmLCD labeled frames: train. last nFrmLCD labeled frames:
        % test
        iFrm = find(tf);
        iFrm = iFrm(:)';
        iTrnAcc = [iTrnAcc iFrm(1:nTrnTst)]; %#ok<AGROW>
        iTstAcc = [iTstAcc iFrm(end-nTrnTst+1:end)]; %#ok<AGROW>
      end
      
      obj.iTrn = iTrnAcc;
      obj.iTst = iTstAcc;      
    end
        
    function hFig = vizWithFurthestFirst(obj)
      % Display furthestfirst plot for all training data (.iTrn);
      % clicking on plot shows training shapes in that vicinity
      % clickingfor groups in subplots; enable
      % clicking on subplots to visualize training shape
      
      hFig = figure;
      
      % use furthestfirst to order shapes by decreasing distance
      warnst = warning('off','backtrace');
      [~,~,ffdidx,~,ffd] = furthestfirst(obj.pGTTrn,obj.NTrn,'Start',[]);
      warning(warnst);
      ffd(1) = inf;
      assert(isequal(sort(ffd,'descend'),ffd));
      
      plot(ffd);
      grid('on');
      title('Training data. Click to view training data near point.','interpreter','none','fontweight','bold');
      ylabel('Minimum distance to training set (px^2)');
      ax = gca;
      ax.YScale = 'log';
      bdfCbk = {@(x,y)nst(x,y)};
     
      LiveDataCursor(hFig,gca,bdfCbk);
      
      function nst(xsel,ysel) %#ok<INUSD>
        %xsel, ysel: (x,y) on ffd plot nearest to user clic
        NSHOW = 6;
        rad = NSHOW/2;
        i0 = max(1,xsel-rad);
        i1 = min(obj.NTrn,xsel+rad);
        idxShow = obj.iTrn(ffdidx(i0:i1));
        obj.vizIdx(idxShow);
      end
    end
    
    function [iTrn,iTstAll,iTstLbl] = genITrnITstJan(obj,idTest)
      % Given experiment id (testID), generate training and test exps.
      %
      % Abstract specification of how this works:
      % * All rows of data include 'group' (date-fly) and 'file' (id, or
      % movie) lbls
      % * For all groups EXCEPT the group of idTest, we include all labeled
      % data with a furthestfirst distance greater than a threshold;
      % except, no group can be overrepresented relative to another by a
      % certain factor.
      % * For the group of idTest, we include all labeled data with a
      % furthestfirst distance greater than a threshold, except we do not
      % included any data for the file/id itTest itself.
           
      assert(obj.nView==1);

      dfTest = idTest(1:9);
      
      tfLbled = obj.isFullyLabeled;
      obj.janExpandMDTable();
      tMD = obj.MD;
      
      dfs = tMD.datefly;
      dfsUn = unique(dfs);
      dfsUnOther = setdiff(dfsUn,dfTest);
      nDFSUnOther = numel(dfsUnOther);
      
      % number of labeled frames for each dfsUnOther
      dfsUnOtherLbledCnt = cellfun(@(x)nnz(strcmp(dfs,x) & tfLbled),dfsUnOther); 
      disp([{'id' 'nLbledFrm'};[dfsUnOther num2cell(dfsUnOtherLbledCnt)]]);
      minLbledCntDfsUnOther = min(dfsUnOtherLbledCnt);

      fprintf('dfTest: %s. %d other DFs. minLbledCntDfsUnOther: %d.\n',...
        dfTest,nDFSUnOther,minLbledCntDfsUnOther);
      
      % For each dfsUnOther, pick training set.
      %
      % We use all frames with a distance-to-other-frames of at least this
      % threshold, (per furthestfirst()).
      MINDISTACCEPT = 17.0; % in squared pixels I think
      % ... Except, we also do not allow any experiment to be way 
      % overrepresented in the data relative to another, per this ratio.
      MAXDF_POPULATION_RATIO = 3.0;

      % First find all frames for each DFOther that exceed threshold
      iAvailDFOther = cell(nDFSUnOther,1);
      for iDFOther = 1:nDFSUnOther
        df = dfsUnOther{iDFOther};
        iDF = find(strcmp(df,dfs) & tfLbled);
        pDF = obj.pGT(iDF,:);
        nDF = numel(iDF);
        
        % use furthestfirst to order shapes by decreasing distance
        warnst = warning('off','backtrace');
        [~,~,tmpidx,~,mindists] = furthestfirst(pDF,nDF,'Start',[]);  
        warning(warnst);
        
        mindists(1) = inf;
        assert(isequal(sort(mindists,'descend'),mindists));
        tfAcc = mindists > MINDISTACCEPT;
        
        iTrnDF = iDF(tmpidx(tfAcc));
        iAvailDFOther{iDFOther} = iTrnDF;       
        fprintf(1,' ... furthestfirst done for %s. %d/%d trials fall under mindist threshold: %.3f.\n',...
          df,nnz(tfAcc),nDF,MINDISTACCEPT);
      end
      nAvailDFOther = cellfun(@numel,iAvailDFOther);
      
      % Now, apply MAXDF_POPULATION_RATIO limit
      maxNDFTrn = round(min(nAvailDFOther) * MAXDF_POPULATION_RATIO);
      fprintf(1,' Maximum ntrials accepted in any DF: %d\n',maxNDFTrn);
      iTrnDFOther = cell(size(iAvailDFOther));
      for iDFOther = 1:nDFSUnOther
        df = dfsUnOther{iDFOther};

        iAvail = iAvailDFOther{iDFOther};
        nAvail = numel(iAvail);
        nKeep = min(nAvail,maxNDFTrn);
        iTrnDFOther{iDFOther} = iAvail(1:nKeep); % iTrns should be sorted in order of descending distance
        
        fprintf(1,'%s: Using %d/%d trials.\n',df,nKeep,nAvail);
      end
      
      %%% For testDF itself, use all frames better than threshold
      iDF = find(strcmp(dfTest,dfs) & tfLbled & ~strcmp(idTest,tMD.id));
      pDF = obj.pGT(iDF,:);
      nDF = numel(iDF);

      if nDF==0
        iTrnDFTest = [];
        fprintf(1,'No trials for datefly %s that are not for ID %s.\n',... 
          dfTest,idTest);
      else
        warnst = warning('off','backtrace');
        [~,~,tmpidx,~,mindists] = furthestfirst(pDF,nDF,'Start',[]);  
        warning(warnst);        
        mindists(1) = inf;
        assert(isequal(sort(mindists,'descend'),mindists));
        tfAcc = mindists > MINDISTACCEPT;

        iTrnDFTest = iDF(tmpidx(tfAcc));
        fprintf(1,'Using %d/%d from datefly %s (but not id %s)\n',...
          numel(iTrnDFTest),numel(iDF),dfTest,idTest);
      end
      
      iTrn = cat(1,iTrnDFOther{:},iTrnDFTest);
      iTstAll = find(strcmp(idTest,tMD.id));
      fprintf(1,'id %s: %d frames for iTstAll.\n',idTest,numel(iTstAll));
      iTstLbl = find(strcmp(idTest,tMD.id) & tfLbled);
      fprintf(1,'id %s: %d frames for iTstLbl.\n',idTest,numel(iTstLbl));
    end
    
    function vizITrnITst(obj,iTrn,iTstAll,iTstLbl)
      % Summarize/visualize iTrn/etc
      
      assert(obj.nView==1);

      fprintf(2,'Summary: iTrn\n');
      obj.summarize(iTrn);
      fprintf(2,'Summary: iTstAll\n');
      obj.summarize(iTstAll);
      fprintf(2,'Summary: iTstLbl\n');
      obj.summarize(iTstLbl);
      
      dfTrn = obj.MD.datefly(iTrn);
      dfTrnUn = unique(dfTrn);
      nDFTrnUn = numel(dfTrnUn);
      
      figure;
      axDistribs = createsubplots(2,3,.07); % axes for pairwise-distance distributions
      axDistribs = reshape(axDistribs,2,3);
      figure;
      axImSimilar = createsubplots(2,nDFTrnUn);  % axes for images
      axImSimilar = reshape(axImSimilar,2,nDFTrnUn);
      figure;
      axImDiff = createsubplots(2,nDFTrnUn);  
      axImDiff = reshape(axImDiff,2,nDFTrnUn);
      for iDF = 1:nDFTrnUn
        df = dfTrnUn{iDF};
        
        tf = strcmp(df,dfTrn);
        iTrnDF = iTrn(tf);
        
        pDFTrn = obj.pGT(iTrnDF,:); 
        distmat = dist2(pDFTrn,pDFTrn);
        tfTriu = logical(triu(ones(size(distmat)),1));
        dists = distmat(tfTriu); % all pairwise distances
        ndists = numel(dists);
        assert(ndists==numel(iTrnDF)*(numel(iTrnDF)-1)/2);
        mudist = mean(dists);
        dists = sort(dists,'descend');  

        axDF = axDistribs(iDF);
        plot(axDF,1:ndists,sort(dists,'descend'),'.','MarkerSize',8);
        hold(axDF,'on');
        plot(axDF,[1 ndists],[mudist mudist],'r');
        grid(axDF,'on');
        tstr = sprintf('%s: mu=%.3f',df,mudist);
        title(axDF,tstr,'interpreter','none','fontweight','bold');        
        
        [iDiff,jDiff] = find(distmat==dists(1) & tfTriu,1);
        [iSim,jSim] = find(distmat==dists(end) & tfTriu,1);
        
        iDiff = iTrnDF([iDiff jDiff]);
        iSim = iTrnDF([iSim jSim]);        
        colors = jet(obj.nfids);
        
        % iSim: plot
        pSim1 = obj.pGT(iSim(1),:);
        pSim2 = obj.pGT(iSim(2),:);
        tstrSim1 = sprintf('%s: frm%04d. dist=%.3f',...
          obj.MD.id{iSim(1)},obj.MD.frm(iSim(1)),dists(end));
        tstrSim2 = sprintf('%s: frm%04d',obj.MD.id{iSim(2)},obj.MD.frm(iSim(2)));
        axSim1 = axImSimilar(1,iDF);
        axSim2 = axImSimilar(2,iDF);
        imagesc(obj.I{iSim(1)},'Parent',axSim1,[0,255]);
        imagesc(obj.I{iSim(2)},'Parent',axSim2,[0,255]);        
        colormap(axSim1,'gray');
        colormap(axSim2,'gray');
        axis(axSim1,'image','off');
        axis(axSim2,'image','off');
        hold(axSim1,'on');
        title(axSim1,tstrSim1,'interpreter','none','fontweight','bold');
        title(axSim2,tstrSim2,'interpreter','none','fontweight','bold');
        for j = 1:obj.nfids
          plot(axSim1,pSim1(j),pSim1(j+obj.nfids),...
            'wo','MarkerFaceColor',colors(j,:));
          plot(axSim1,pSim2(j),pSim2(j+obj.nfids),...
            'ws','MarkerFaceColor',colors(j,:));
        end
        
        % iDiff: plot
        pDiff1 = obj.pGT(iDiff(1),:);
        pDiff2 = obj.pGT(iDiff(2),:);
        tstrDiff1 = sprintf('%s: frm%04d. dist=%.3f',...
          obj.MD.id{iDiff(1)},obj.MD.frm(iDiff(1)),dists(1));
        tstrDiff2 = sprintf('%s: frm%04d',obj.MD.id{iDiff(2)},obj.MD.frm(iDiff(2)));
        axDiff1 = axImDiff(1,iDF);
        axDiff2 = axImDiff(2,iDF);
        imagesc(obj.I{iDiff(1)},'Parent',axDiff1,[0,255]);
        imagesc(obj.I{iDiff(2)},'Parent',axDiff2,[0,255]);        
        colormap(axDiff1,'gray');
        colormap(axDiff2,'gray');
        axis(axDiff1,'image','off');
        axis(axDiff2,'image','off');
        hold(axDiff1,'on');
        title(axDiff1,tstrDiff1,'interpreter','none','fontweight','bold');
        title(axDiff2,tstrDiff2,'interpreter','none','fontweight','bold');
        for j = 1:obj.nfids
          plot(axDiff1,pDiff1(j),pDiff1(j+obj.nfids),...
            'wo','MarkerFaceColor',colors(j,:));
          plot(axDiff1,pDiff2(j),pDiff2(j+obj.nfids),...
            'ws','MarkerFaceColor',colors(j,:));
        end
        
        linkaxes(axImSimilar(:,iDF));
        linkaxes(axImDiff(:,iDF));        
      end
      
      linkaxes(axDistribs,'y');   
    end
    
    function [iTrn,iTstAll,iTstLbl] = genITrnITst1(obj,exp)
      % Given experiment name (exp), generate training and test experiments.
      %
      % exp: full experiment name, eg 150723_2_002_4_xxxx.      
      
      sExp = FS.parseexp(exp);
      tfIsFullyLabeled = obj.isFullyLabeled;
      obj.janExpandMDTable();
      tMD = obj.MD;
      
      dfs = tMD.datefly;
      dfsUn = unique(dfs);
      dfsUnOther = setdiff(dfsUn,sExp.datefly);
      nDfsUnOther = numel(dfsUnOther);
      
      % For each datefly-other (not selected datefly), try to get ~400
      % lblact frames, half as many lblrest frames
      LBLACT_FRAMES_PERDATEFLY = 400;
      LBLRST_FRAMES_PERDATEFLY = 200;
      iTrlAct = zeros(0,1);
      iTrlRst = zeros(0,1);
      for iDF = 1:nDfsUnOther
        df = dfsUnOther{iDF};
        
        [idsUn,idsNLblRstAvail,idsNLblActAvail] = lclGetExpsForDateFly(tMD,df);
        nIdsUn = numel(idsUn);
        fprintf('Working on datefly %s. %d exps.\n',df,nIdsUn);
        
        idsNLblRstTake = loadBalance(...
          min(LBLRST_FRAMES_PERDATEFLY,sum(idsNLblRstAvail)),idsNLblRstAvail);
        idsNLblActTake = loadBalance(...
          min(LBLACT_FRAMES_PERDATEFLY,sum(idsNLblActAvail)),idsNLblActAvail);
        for iID = 1:nIdsUn
          id = idsUn{iID};          
          [iTmpLblRst,iTmpLblAct] = lclGetActRstFrms(tMD,tfIsFullyLabeled,id,...
            idsNLblRstTake(iID),idsNLblActTake(iID)); 
          iTrlRst = [iTrlRst;iTmpLblRst]; %#ok<AGROW>
          iTrlAct = [iTrlAct;iTmpLblAct]; %#ok<AGROW>
        end
      end
      
      % for this/selected datefly, take maximum roughly balanced number of
      % actives, and half that many rests. Exclude specified experiment
      % though
      df = sExp.datefly;
      [idsUn,idsNLblRstAvail,idsNLblActAvail] = lclGetExpsForDateFly(tMD,df);
      tfTmp = strcmp(idsUn,sExp.id);
      if any(tfTmp)
        assert(nnz(tfTmp)==1);
        fprintf(1,'Selected datefly is present in dataset; will not include in training set.\n');
      end
      idsUn = idsUn(~tfTmp,:);
      idsNLblRstAvail = idsNLblRstAvail(~tfTmp,:);
      idsNLblActAvail = idsNLblActAvail(~tfTmp,:);   
      
      if isempty(idsUn)
        fprintf(1,'No other ids for selected datefly.\n');
      else      
        nIdsUn = numel(idsUn);
        fprintf('Working on SELECTED datefly %s. %d exps.\n',df,nIdsUn);
        FUDGEFAC = 1.4; % try to get as much data as possible
        nActTot = min(idsNLblActAvail)*FUDGEFAC*nIdsUn;
        nRstTot = round(nActTot/2);
        idsNLblRstTake = loadBalance(min(nRstTot,sum(idsNLblRstAvail)),idsNLblRstAvail);
        idsNLblActTake = loadBalance(min(nActTot,sum(idsNLblActAvail)),idsNLblActAvail);

        for iID = 1:nIdsUn
          id = idsUn{iID};
          [iTmpLblRst,iTmpLblAct] = lclGetActRstFrms(tMD,tfIsFullyLabeled,id,...
            idsNLblRstTake(iID),idsNLblActTake(iID));
          iTrlRst = [iTrlRst;iTmpLblRst]; %#ok<AGROW>
          iTrlAct = [iTrlAct;iTmpLblAct]; %#ok<AGROW>
        end
      end
      
      iTrn = [iTrlAct;iTrlRst];
      tfID = strcmp(sExp.id,tMD.id);
      iTstAll = find(tfID);
      iTstLbl = find(tfIsFullyLabeled & tfID);
      fprintf(1,'nTrn nTstAll nTstLbl %d %d %d\n',numel(iTrn),numel(iTstAll),numel(iTstLbl));
    end
    
    function janExpandMDTable(obj)
      tMD = obj.MD;
      
      if ~ismember('datefly',tMD.Properties.VariableNames)
        fprintf(1,'Augmenting .MD table.\n');
        s = cellfun(@FS.parseexp,tMD.lblFile);
        tMD2 = struct2table(s);        
        assert(isequal(tMD.lblFile,tMD2.orig));
        tMD = [tMD tMD2];
        obj.MD = tMD;
      end
      obj.MD = tMD;
    end
  end
  
end

function [idsUn,nLblRstAvail,nLblActAvail] = lclGetExpsForDateFly(tMD,df)
% Get ids/metadata for all exps for a given date-fly

tfDF = strcmp(tMD.datefly,df);

idsUn = unique(tMD.id(tfDF));
nIdsUn = numel(idsUn);
nLblRstAvail = nan(size(idsUn));
nLblActAvail = nan(size(idsUn));
for iID = 1:nIdsUn
  id = idsUn{iID};
  tfID = strcmp(tMD.id,id);
  
  tmp = unique(tMD.nlblrest(tfID));
  assert(isscalar(tmp));
  nLblRstAvail(iID) = tmp;
  
  tmp = unique(tMD.nlblactv(tfID));
  assert(isscalar(tmp));
  nLblActAvail(iID) = tmp;
end
end

function [iLblRst,iLblAct] = lclGetActRstFrms(tMD,tfLbled,id,nRst,nAct)
% Get labeled active/rest frames for given id
%
% tMD: [NxM] metadata table
% tfLbled: [N] logical, eg .isFullyLabeled
% id: id
% nAct/nRst: numbers of active/resting frames to get

tfID = strcmp(tMD.id,id);

nFrmID = nnz(tfID);
nFrmLblID = nnz(tfID & tfLbled);
actvf0 = unique(tMD.actvf0(tfID));
actvf1 = unique(tMD.actvf1(tfID));
assert(isscalar(actvf0) && isscalar(actvf1));

tfLblAct = tfID & tfLbled & (tMD.frm>=actvf0 & tMD.frm<=actvf1);
tfLblRst = tfID & tfLbled & ~(tMD.frm>=actvf0 & tMD.frm<=actvf1);
iLblAct = find(tfLblAct);
iLblRst = find(tfLblRst);
fprintf(1,'  ID: %s, %d frm, %d lblfrm. acvtf: [%d %d]. nrest nact: %d %d... ',id,...
  nFrmID,nFrmLblID,actvf0,actvf1,numel(iLblRst),numel(iLblAct));

iLblRst = randsample(iLblRst,nRst);
iLblAct = randsample(iLblAct,nAct);
fprintf(1,' ...adding %d lblrst, %d lblact frames\n',...
  numel(iLblRst),numel(iLblAct));
end