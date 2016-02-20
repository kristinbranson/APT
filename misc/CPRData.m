classdef CPRData < handle
  properties (Constant)
    % for resting/active ranges etc
    MD_XLSFILE = 'f:\DropBoxNEW\DropBox\Tracking_KAJ\track.results\summary.xlsx';
    MD_XLSSHEET = 'gt data';      
  end
  
  properties
    Name    % Name of this CPRData
    MD      % [NxR] Table of Metadata
    
    I       % [N] column cell vec, images
    pGT     % [NxD] GT shapes for I
    bboxes  % [Nx2d] bboxes for I 
    
    Ipp     % [N] cell vec of preprocessed 'channel' images. .iPP{i} is [nrxncxnchan]
    IppInfo % [nchan] cellstr describing each channel (3rd dim) of .Ipp
    
    H0      % [256x1] for histeq
    
    iTrn    % [Ntrn] indices into I for training set
    iTst    % [Ntst] indices into I for test set
  end
  properties (Dependent)
    N
    d
    D
    nfids
    
    isLabeled % [Nx1] logical, whether trial N has labels    
    isFullyLabeled % [Nx1] logical
    NFullyLabeled

    iUnused % trial indices (1..N) that are in neither iTrn or iTst
    
    NTrn
    NTst
    ITrn
    ITst
    pGTTrn
    pGTTst
    bboxesTrn
    bboxesTst
    MDTrn
    MDTst
  end
  methods 
    function v = get.N(obj)
      v = numel(obj.I);
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
  methods
    
    function obj = CPRData(varargin)
      % obj = CPRData(movFiles)
      % obj = CPRData(lblFiles,tfAllFrames)
      % obj = CPRData(Is,p,bb,md)
      
      switch nargin
        case 1
          movFiles = varargin{1};
          [Is,bb,md] = CPRData.readMovs(movFiles);
          p = nan(size(Is,1),0);
        case 2
          lblFiles = varargin{1};
          tfAllFrames = varargin{2};
          [Is,p,bb,md] = CPRData.readLblFiles(lblFiles,'tfAllFrames',tfAllFrames);
        case 3
          [Is,p,bb] = deal(varargin{:});
        case 4
          [Is,p,bb,md] = deal(varargin{:});
        otherwise
          assert(false,'Invalid constructor args.');
      end
      
      assert(iscolumn(Is) && iscell(Is));
      N = numel(Is);
      assert(size(p,1)==N);
      if iscell(bb)
        bb = cat(1,bb{:});
      end
      assert(size(bb,1)==N);  
      if exist('md','var')==0
        md = cell2table(cell(N,0));
      end

      obj.MD = md;      
      obj.I = Is;
      obj.pGT = p;
      obj.bboxes = bb;
    end
        
    function [sgscnts,slscnts,sgsedge,slsedge] = calibIpp(obj,nsamp)
      % Sample SGS/SLS intensity histograms 
      %
      % nsamp: number of trials to sample
      
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
    function [sgsthresh,slsspan] = calibIpp2(sgscnts,slscnts,sgsedge,slsedge)
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
  methods
    function computeIpp(obj,varargin)
      % Preprocess images and set .Ipp.
      % sig1,sig2: see Features.pp
      % Optional PVs:
      % - iChan index vectors into channels for which channels to
      % keep/store.
      % - iTrl. trial indices for which Ipp should be computed. Defaults to
      % find(obj.isFullyLabeled).
      % - See Features.pp for other optional pvs

      iTrl = myparse(varargin,'iTrl',find(obj.isFullyLabeled));
      nTrl = numel(iTrl);
      
      sig1 = [0 2 4 8]; % for now, normalizing/rescaling channels assuming these sigs
      sig2 = [0 2 4 8];
      iChan = [...
        2 3 ... % blur sig1(1:3)
        5 6 7 9 10 11 13 14 17 ... % SGS
        22 23 25 26 27 29 30]; % SLS 
      
      [S,SGS,SLS] = Features.pp(obj.I(iTrl),sig1,sig2); %,varargin{:});
      
      n1 = numel(sig1);
      n2 = numel(sig2);
      assert(iscell(S) && isequal(size(S),[nTrl n1]));
      assert(iscell(SGS) && isequal(size(SGS),[nTrl n1 n2]));
      assert(iscell(SLS) && isequal(size(SLS),[nTrl n1 n2]));

      ipp = cell(nTrl,1);
      for i = 1:nTrl
        ipp{i} = cat(3,S{i,:},SGS{i,:},SLS{i,:});
        assert(size(ipp{i},3)==n1+n1*n2+n1*n2);        
      end
      ippInfo = arrayfun(@(x)sprintf('S:sig1=%.2f',x),sig1(:),'uni',0);
      for i2 = 1:n2 % note raster order corresponding to order of ipp{iTrl}
        for i1 = 1:n1
          ippInfo{end+1,1} = sprintf('SGS:sig1=%.2f,sig2=%.2f',sig1(i1),sig2(i2)); %#ok<AGROW>
        end
      end
      for i2 = 1:n2 % etc
        for i1 = 1:n1
          ippInfo{end+1,1} = sprintf('SLS:sig1=%.2f,sig2=%.2f',sig1(i1),sig2(i2)); %#ok<AGROW>
        end
      end
      
      ipp = cellfun(@(x)x(:,:,iChan),ipp,'uni',0);
      ippInfo = ippInfo(iChan);
      obj.Ipp = cell(obj.N,1);
      obj.Ipp(iTrl) = ipp;
      obj.IppInfo = ippInfo;
    end
    
    function varargout = viz(obj,varargin)
      [varargout{1:nargout}] = Shape.viz(obj.I(obj.isFullyLabeled,:),obj.pGT(obj.isFullyLabeled,:),...
        struct('nfids',obj.nfids,'D',obj.D),'md',obj.MD(obj.isFullyLabeled,:),varargin{:});
    end
    
    function n = getFilename(obj)
      n = sprintf('td_%s_%s.mat',obj.Name,datestr(now,'yyyymmdd'));
    end
    
    function append(obj,varargin)
      % cat/append additional CPRDatas
      
      warning('CPRData:append','Clearing H0.');
      obj.H0 = [];
      
      for i = 1:numel(varargin)
        dd = varargin{i};
        assert(isequal(dd.IppInfo,obj.IppInfo),'Different IppInfo found for data index %d.',i);
        
        Nbefore = numel(obj.I); 
        
        obj.MD = cat(1,obj.MD,dd.MD);
        obj.I = cat(1,obj.I,dd.I);
        obj.pGT = cat(1,obj.pGT,dd.pGT);
        obj.bboxes = cat(1,obj.bboxes,dd.bboxes);
        obj.Ipp = cat(1,obj.Ipp,dd.Ipp);

        obj.iTrn = cat(2,obj.iTrn,dd.iTrn+Nbefore);
        obj.iTst = cat(2,obj.iTst,dd.iTst+Nbefore);
      end
      
      ids = strcat(obj.MD.lblFile,'#',...
        strtrim(cellstr(num2str(obj.MD.iMov))),'#',strtrim(cellstr(num2str(obj.MD.frm))));
      assert(numel(ids)==numel(unique(ids)),'Duplicate frame metadata encountered.');
      
      [lblFileUn,~,idx] = unique(obj.MD.lblFile);
      obj.MD.iLbl = idx;
      fprintf(1,'Relabeled MD.iLbl:\n');
      disp([lblFileUn num2cell((1:numel(lblFileUn))')]);
    end
    
    function summarize(obj,iTrl)
      % iTrl: vector of trial indices
      
      tMD = obj.MD(iTrl,:);
      tfLbled = obj.isFullyLabeled(iTrl,:);
      
      dfUn = unique(tMD.datefly);
      nDF = numel(dfUn);
      for iDF = 1:nDF
        df = dfUn{iDF};
        tfDF = strcmp(tMD.datefly,df);
        fprintf(1,'datefly: %s\n',df);
        
        idsUn = unique(tMD.id(tfDF));
        nID = numel(idsUn);
        for iID = 1:nID
          id = idsUn{iID};
          tfID = strcmp(tMD.id,id);
          tfIDLbled = strcmp(tMD.id,id) & tfLbled;
          
          nLblAct = nnz(tMD.tfAct(tfIDLbled));
          nLblRst = nnz(tMD.tfRst(tfIDLbled));
          fprintf(1,'id %s. nfrm=%d. nfrmLbled=%d, nRst/nAct=%d/%d.\n',...
            id,nnz(tfID),nnz(tfIDLbled),nLblRst,nLblAct);
        end
      end
    end

  end
  
  methods (Static)
    
    function [I,p,bb,md] = readLblFiles(lblFiles,varargin)
      % lblFiles: [N] cellstr
      % Optional PVs:
      %  - tfAllFrames. scalar logical, defaults to false. If true, read in
      %  unlabeled as well as labeled frames.
      % 
      % I: [Nx1] cell array of images (frames)
      % p: [NxD] positions
      % bb: [Nx2d] bboxes
      % md: [Nxm] metadata table

      assert(iscellstr(lblFiles));
      nLbls = numel(lblFiles);

      tfAllFrames = myparse(varargin,'tfAllFrames',false);

      mr = MovieReader();
      I = cell(0,1);
      p = [];
      sMD = struct('iLbl',cell(0,1),'lblFile',[],'iMov',[],...
        'iFrm',[],'frm',[],'nTag',[],'tagPtsBinVec',[]);
      
      for iLbl = 1:nLbls
        lblName = lblFiles{iLbl};
        lbl = load(lblName,'-mat');
        nMov = numel(lbl.movieFilesAll);
        fprintf('Lbl file: %s, %d movies.\n',lblName,nMov);
        
        for iMov = 1:nMov
          movName = lbl.movieFilesAll{iMov};
          mr.open(movName);
          
          lpos = lbl.labeledpos{iMov}; % npts x 2 x nframes
          lpostag = lbl.labeledpostag{iMov};
          assert(size(lpostag,2)==size(lpos,3));
          [npts,d,nFrmAll] = size(lpos);
          D = d*npts; % d*npts;

          % find labeled/tagged frames
          frmsLbled = find(~isnan(lpos(1,1,:))); % labeled frames
          tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
          ntagged = sum(tftagged,1);
          frmsTagged = find(ntagged);
          assert(all(ismember(frmsTagged,frmsLbled)));
          nFrmsLbled = numel(frmsLbled);
          nFrmsTagged = numel(frmsTagged);
          
          if tfAllFrames
            nFrmRead = nFrmAll; % number of frames to read for this iLbl/iMov
            frms2Read = 1:nFrmRead;
          else          
            nFrmRead = nFrmsLbled;
            frms2Read = frmsLbled;
          end
          ITmp = cell(nFrmRead,1);
          pTmp = nan(nFrmRead,D);
          fprintf('  mov %d, D=%d, reading %d frames (%d labeled frames, %d tagged frames)\n',...
            iMov,D,nFrmRead,nFrmsLbled,nFrmsTagged);
          
          for iFrm = 1:nFrmRead
            f = frms2Read(iFrm);
            im = mr.readframe(f);
            
            fprintf('iLbl=%d, iMov=%d, read frame %d (%d/%d)\n',...
              iLbl,iMov,f,iFrm,nFrmRead);
            
            ITmp{iFrm} = im;
            lblsFrmXY = lpos(:,:,f);
            pTmp(iFrm,:) = Shape.xy2vec(lblsFrmXY);
            
            tagbinvec = sum(tftagged(:,f)'.*2.^(0:npts-1));
            sMD(end+1,1).iLbl = iLbl; %#ok<AGROW>
            sMD(end).lblFile = lblName;
            sMD(end).iMov = iMov;
            sMD(end).iFrm = iFrm;
            sMD(end).frm = f;
            sMD(end).nTag = ntagged(f);
            sMD(end).tagPtsBinVec = tagbinvec;
          end
          
          I = [I;ITmp]; %#ok<AGROW>
          p = [p;pTmp]; %#ok<AGROW>
        end
      end
      
      sz = cellfun(@(x)size(x'),I,'uni',0);
      bb = cellfun(@(x)[[1 1] x],sz,'uni',0);

      assert(isequal(numel(sMD),numel(I),size(p,1),size(bb,1)));
      md = struct2table(sMD);
    end
    
    function [I,bb,md] = readMovs(movFiles)
      % movFiles: [N] cellstr
      % Optional PVs:
      % 
      % I: [Nx1] cell array of images (frames)
      % bb: [Nx2d] bboxes
      % md: [Nxm] metadata table

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
    
  end
  
  methods % histeq
  
    function vizHist(obj,varargin)
      smoothspan = myparse(varargin,'smoothspan',nan);
      tfsmooth = ~isnan(smoothspan);
      
      H = nan(256,obj.N);
      for i = 1:obj.N
        H(:,i) = imhist(obj.I{i});
      end
      nLbl = numel(unique(obj.MD.iLbl));
      muLbl = nan(256,nLbl);
      sdLbl = nan(256,nLbl);
      for iLbl = 1:nLbl
        tf = obj.MD.iLbl==iLbl;
        Hlbl = H(:,tf);
        fprintf('Working on iLbl=%d, n=%d.\n',iLbl,nnz(tf));
        muLbl(:,iLbl) = nanmean(Hlbl,2);
        sdLbl(:,iLbl) = nanstd(Hlbl,[],2);
      end
      
      if tfsmooth
        for iLbl = 1:nLbl
          muLbl(:,iLbl) = smooth(muLbl(:,iLbl),smoothspan);
          sdLbl(:,iLbl) = smooth(sdLbl(:,iLbl),smoothspan);
        end
      end
      
      figure;
      x = 1:256;
      plot(x,muLbl,'linewidth',2);
      legend(arrayfun(@num2str,1:nLbl,'uni',0));
      hold on;
      ax = gca;
      ax.ColorOrderIndex = 1;
      plot(x,muLbl-sdLbl);
      ax.ColorOrderIndex = 1;
      plot(x,muLbl+sdLbl);
      
      grid on;
    end
    
    function H0 = histEq(obj,H0)
      tfH0Given = exist('H0','var')>0;
      
      imSz = cellfun(@size,obj.I,'uni',0);
      cellfun(@(x)assert(isequal(x,imSz{1})),imSz);
      imSz = imSz{1};
      imNpx = numel(obj.I{1});
        
      if tfH0Given
        % none
      else
        H = nan(256,obj.N);
        mu = nan(1,obj.N);
        for iTrl = 1:obj.N
          im = obj.I{iTrl};
          H(:,iTrl) = imhist(im);
          mu(iTrl) = mean(im(:));
        end
        % normalize to brighter movies, not to dimmer movies
        idxuse = mu >= prctile(mu,75);
        fprintf('using %d frames to form H0:\n',numel(idxuse));
        H0 = median(H(:,idxuse),2);
        H0 = H0/sum(H0)*imNpx;
      end
      obj.H0 = H0;

      % normalize one video at a time
      warning('TD:histeq','Currently assuming one movie per lbl.');
      nLbl = numel(unique(obj.MD.iLbl));
      for iLbl = 1:nLbl
        tfLbl = obj.MD.iLbl==iLbl;
        % iTrlLbl = idx(ld.expidx(idx)==iLbl);
        if ~any(tfLbl)
          continue;
        end
        bigim = cat(1,obj.I{tfLbl});
        bigimnorm = histeq(bigim,H0);
        obj.I(tfLbl) = mat2cell(bigimnorm,...
          repmat(imSz(1),[nnz(tfLbl) 1]),imSz(2));
      end
    end
      
  end
  
  methods % partitions
    
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
    
    function [iTrn,iTstAll,iTstLbl] = genITrnITst1(obj,exp)
      % Given experiment name (exp), generate training and test experiments.
      %
      % exp: full experiment name, eg 150723_2_002_4_xxxx.      
      
      sExp = FS.parseexp(exp);
      tfIsFullyLabeled = obj.isFullyLabeled;
      obj.expandMDTable();
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
    
    function expandMDTable(obj)
      tMD = obj.MD;
      
      if ~ismember('datefly',tMD.Properties.VariableNames)
        fprintf(1,'Augmenting .MD table.\n');
        s = cellfun(@FS.parseexp,tMD.lblFile);
        tMD2 = struct2table(s);        
        assert(isequal(tMD.lblFile,tMD2.orig));
        tMD = [tMD tMD2];
        obj.MD = tMD;
      end
      if ~ismember('actvf0',tMD.Properties.VariableNames)
        tXLS = readtable(obj.MD_XLSFILE,'Sheet',obj.MD_XLSSHEET);
        tf = ~cellfun(@isempty,tXLS.vid);
        tXLS = tXLS(tf,:);
        
        tMD = join(tMD,tXLS); % should be joining on id)
        
        tMD.tfAct =  (tMD.frm>=tMD.actvf0 & tMD.frm<=tMD.actvf1);
        tMD.tfRst = ~(tMD.frm>=tMD.actvf0 & tMD.frm<=tMD.actvf1);
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