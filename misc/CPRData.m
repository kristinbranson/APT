classdef CPRData < handle
  
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
  
  %% Dep prop getters
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
  
  %% Ctor
  methods
    
    function obj = CPRData(varargin)
      % obj = CPRData(movFiles)
      % obj = CPRData(lblFiles,tfAllFrames)
      % obj = CPRData(I,tblP)
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
            lblFiles = varargin{1};
            tfAllFrames = varargin{2};
            [Is,p,md] = CPRData.readLblFiles(lblFiles,'tfAllFrames',tfAllFrames);
          end
          sz = cellfun(@(x)size(x'),Is,'uni',0);
          bb = cellfun(@(x)[[1 1] x],sz,'uni',0);
        otherwise % 3+
          [movFiles,lposes,lpostags] = deal(varargin{1:3});
          if ischar(varargin{4}) && any(strcmp(varargin{4},{'all' 'lbl'}))
            type = varargin{4};
            varargin = varargin(5:end);
            [Is,tbl] = CPRData.readMovsLbls(movFiles,lposes,lpostags,type,varargin{:});
          else
            iMovs = varargin{4};
            frms = varargin{5};
            varargin = varargin(6:end);
            [Is,tbl] = CPRData.readMovsLblsRaw(movFiles,lposes,lpostags,iMovs,frms,varargin{:});
          end
          p = tbl.p;
          tbl(:,'p') = [];
          md = tbl;
          
          sz = cellfun(@(x)size(x'),Is,'uni',0);
          bb = cellfun(@(x)[[1 1] x],sz,'uni',0);
      end
      
      assert(iscolumn(Is) && iscell(Is));
      N = numel(Is);
      assert(size(p,1)==N);
      if iscell(bb)
        bb = cat(1,bb{:});
      end
      assert(size(bb,1)==N);  

      obj.MD = md;      
      obj.I = Is;
      obj.pGT = p;
      obj.bboxes = bb;
    end
    
    function append(obj,varargin)
      % Cat/append additional CPRDatas
      % 
      % data.append(data1,data2,...)

      for i = 1:numel(varargin)
        dd = varargin{i};
        assert(isequaln(dd.H0,obj.H0),'Different H0 found for data index %d.',i);
        assert(isequal(dd.IppInfo,obj.IppInfo),...
          'Different IppInfo found for data index %d.',i);
        
        Nbefore = numel(obj.I);
        
        obj.MD = cat(1,obj.MD,dd.MD);
        obj.I = cat(1,obj.I,dd.I);
        obj.pGT = cat(1,obj.pGT,dd.pGT);
        obj.bboxes = cat(1,obj.bboxes,dd.bboxes);
        obj.Ipp = cat(1,obj.Ipp,dd.Ipp);
        
        obj.iTrn = cat(2,obj.iTrn,dd.iTrn+Nbefore);
        obj.iTst = cat(2,obj.iTst,dd.iTst+Nbefore);
      end
    end    
    
  end
  
  %% Import
  methods (Static)
    
    function [I,p,md] = readLblFiles(lblFiles,varargin)
      % lblFiles: [N] cellstr
      % Optional PVs:
      %  - tfAllFrames. scalar logical, defaults to false. If true, read in
      %  unlabeled as well as labeled frames.
      % 
      % I: [Nx1] cell array of images (frames)
      % p: [NxD] positions
      % md: [Nxm] metadata table

      assert(iscellstr(lblFiles));
      nLbls = numel(lblFiles);

      tfAllFrames = myparse(varargin,'tfAllFrames',false);
      if tfAllFrames
        readMovsLblsType = 'all';
      else
        readMovsLblsType = 'lbl';
      end
      I = cell(0,1);
      p = [];
      md = [];
      for iLbl = 1:nLbls
        lblName = lblFiles{iLbl};
        lbl = load(lblName,'-mat');
        fprintf('Lblfile: %s\n',lblName);
        
        movFiles = lbl.movieFilesAll;
        
        [ILbl,tMDLbl] = CPRData.readMovsLbls(movFiles,...
          lbl.labeledpos,lbl.labeledpostag,readMovsLblsType);
        pLbl = tMDLbl.p;
        tMDLbl(:,'p') = [];
        
        nrows = numel(ILbl);
        tMDLbl.lblFile = repmat({lblName},nrows,1);
        [~,lblNameS] = myfileparts(lblName);
        tMDLbl.lblFileS = repmat({lblNameS},nrows,1);
        
        I = [I;ILbl]; %#ok<AGROW>
        p = [p;pLbl]; %#ok<AGROW>
        md = [md;tMDLbl]; %#ok<AGROW>
      end
      
      assert(isequal(size(md,1),numel(I),size(p,1),size(bb,1)));
    end
    
    function [I,tbl] = readMovsLbls(movieNames,labeledposes,...
        labeledpostags,type,varargin)
      % convenience signature 
      %
      % type: either 'all' or 'lbl'

      nMov = numel(movieNames);      
      switch type
        case 'all'
          frms = repmat({'all'},nMov,1);
        case 'lbl'
          frms = repmat({'lbl'},nMov,1);
        otherwise
          assert(false);
      end
      [I,tbl] = CPRData.readMovsLblsRaw(movieNames,labeledposes,...
        labeledpostags,1:nMov,frms,varargin{:});
    end
    
    function [I,tbl] = readMovsLblsRaw(...
        movieNames,lposes,lpostags,iMovs,frms,varargin)
      % Read moviefiles with landmark labels
      %
      % movieNames: [N] cellstr of movienames
      % lposes: [N] cell array of labeledpos arrays [nptsx2xnfrms]
      % lpostags: [N] cell array of labeledpostags [nptsxnfrms]      
      % iMovs. [M] indices into movieNames to read.
      % frms. [M] cell array. frms{i} is a vector of frames to read for
      % movie iMovs(i). frms{i} may also be:
      %     * 'all' indicating "all frames" 
      %     * 'lbl' indicating "all labeled frames"      
      %
      % I: [Ntrl] cell vec of images
      % tbl: [NTrl rows] labels/metadata table.
      %
      % Optional PVs:
      % - hWaitBar. Waitbar object
      % - noImg. logical scalar default false. If true, all elements of I
      % will be empty.
      
      [hWB,noImg] = myparse(varargin,...
        'hWaitBar',[],...
        'noImg',false);
      assert(numel(iMovs)==numel(frms));
      for i = 1:numel(frms)
        val = frms{i};
        assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
      end
      
      tfWB = ~isempty(hWB);
      
      assert(iscellstr(movieNames));
      assert(iscell(lposes) && iscell(lpostags));
      assert(isequal(numel(movieNames),numel(lposes),numel(lpostags)));
      
      mr = MovieReader();

      I = [];
      s = struct('mov',cell(0,1),'movS',[],'frm',[],'p',[],'tfocc',[]);
      
      nMov = numel(iMovs);
      fprintf('Reading %d movies.\n',nMov);
      for i = 1:nMov
        iMov = iMovs(i);
        mov = movieNames{iMov};
        [~,movS] = myfileparts(mov);
        lpos = lposes{iMov}; % npts x 2 x nframes
        lpostag = lpostags{iMov};

        [npts,d,nFrmAll] = size(lpos);
        assert(isequal(size(lpostag),[npts nFrmAll]));
        D = d*npts;
        
        mr.open(mov);
        
        % find labeled/tagged frames (considering ALL frames for this
        % movie)
        tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
        frmsLbled = find(tfLbled);
        tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
        ntagged = sum(tftagged,1);
        frmsTagged = find(ntagged);
        assert(all(ismember(frmsTagged,frmsLbled)));

        frms2Read = frms{i};
        if strcmp(frms2Read,'all')
          frms2Read = 1:nFrmAll;
        elseif strcmp(frms2Read,'lbl')
          frms2Read = frmsLbled;
        end
        nFrmRead = numel(frms2Read);
        
        ITmp = cell(nFrmRead,1);
        fprintf('  mov %d, D=%d, reading %d frames\n',iMov,D,nFrmRead);
        
        if tfWB
          hWB.Name = 'Reading movies';
          wbStr = sprintf('Reading movie %s',movS);
          waitbar(0,hWB,wbStr);          
        end
        for iFrm = 1:nFrmRead
          if tfWB
            waitbar(iFrm/nFrmRead,hWB);
          end
          
          f = frms2Read(iFrm);
          if noImg
            im = [];
          else
            im = mr.readframe(f);
            if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
              im = rgb2gray(im);
            end
          end
          
          %fprintf('iMov=%d, read frame %d (%d/%d)\n',iMov,f,iFrm,nFrmRead);
          
          ITmp{iFrm} = im;
          lblsFrmXY = lpos(:,:,f);
          tags = lpostag(:,f);
          
          s(end+1,1).mov = mov; %#ok<AGROW>
          s(end).movS = movS;
          s(end).frm = f;
          s(end).p = Shape.xy2vec(lblsFrmXY);
          s(end).tfocc = strcmp('occ',tags(:)');          
        end
        
        I = [I;ITmp]; %#ok<AGROW>
      end
      tbl = struct2table(s);      
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
    
    function I = getFrames(tbl)
      % Read frames from movies given MD table
      % 
      % tbl: [NxR] metadata table
      % 
      % I: [N] cell vector of images for each row of tbl
      
      N = size(tbl,1);
      movsUn = unique(tbl.mov);
      [~,movUnIdx] = ismember(tbl.mov,movsUn);
      frms = tbl.frm;
      
      % open movies in MovieReaders
      nMovUn = numel(movsUn);
      for iTrl = nMovUn:-1:1
        mrs(iTrl,1) = MovieReader();
        mrs(iTrl).forceGrayscale = true;
        mrs(iTrl).open(movsUn{iTrl});
      end
      
      I = cell(N,1);
      for iTrl = 1:N
        iMov = movUnIdx(iTrl);
        f = frms(iTrl);
        
        mr = mrs(iMov);
        im = mr.readframe(f); % currently forceGrayscale
        I{iTrl} = im;
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
    
    function H0 = histEq(obj,varargin)
      % Perform histogram equalization on all images
      %
      % Optional PVs:
      % H0: [nbin] intensity histogram used in equalization
      % g: [N] grouping vector, either numeric or categorical. Images with
      % the same value of g are histogram-equalized together. For example,
      % g might indicate which movie the image is taken from.
      
      [H0,nbin,g,hWB] = myparse(varargin,...
        'H0',[],...
        'nbin',256,...
        'g',ones(obj.N,1),...
        'hWaitBar',[]);
      tfH0Given = ~isempty(H0);
      tfWB = ~isempty(hWB);
      
      imSz = cellfun(@size,obj.I,'uni',0);
      cellfun(@(x)assert(isequal(x,imSz{1})),imSz);
      imSz = imSz{1};
      imNpx = numel(obj.I{1});
      
      if tfH0Given
        % none
      else
        H = nan(nbin,obj.N);
        mu = nan(1,obj.N);
        loc = [];
        if tfWB
          waitbar(0,hWB,'Performing histogram equalization...','name','Histogram Equalization');
        end
        for iTrl = 1:obj.N
          if tfWB
            waitbar(iTrl/obj.N,hWB);
          end
          
          im = obj.I{iTrl};
          [H(:,iTrl),loctmp] = imhist(im,nbin);
          if iTrl==1
            loc = loctmp;
          else
            assert(isequal(loctmp,loc));
          end
          mu(iTrl) = mean(im(:));
        end
        % normalize to brighter movies, not to dimmer movies
        tfuse = mu >= prctile(mu,75);
        fprintf('using %d frames to form H0:\n',nnz(tfuse));
        H0 = median(H(:,tfuse),2);
        H0 = H0/sum(H0)*imNpx;
      end
      obj.H0 = H0;
      
      % normalize one group at a time
      gUn = unique(g);
      nGrp = numel(gUn);
      fprintf(1,'%d groups to equalize.\n',nGrp);
      for iGrp = 1:nGrp
        gCur = gUn(iGrp);
        tf = g==gCur;
        
        bigim = cat(1,obj.I{tf});
        bigimnorm = histeq(bigim,H0);
        obj.I(tf) = mat2cell(bigimnorm,...
          repmat(imSz(1),[nnz(tf) 1]),imSz(2));
      end
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
      
      if jan
        fprintf(1,'Using "Jan" settings.\n');
        pause(2);
        sig1 = [0 2 4 8]; % for now, normalizing/rescaling channels assuming these sigs
        sig2 = [0 2 4 8];
        iChan = [...
          2 3 ... % blur sig1(1:3)
          5 6 7 9 10 11 13 14 17 ... % SGS
          22 23 25 26 27 29 30]; % SLS
      end
      if romain
        fprintf(1,'Using "Romain" settings.\n');
        pause(2);
        sig1 = [0 2 4 8]; % for now, normalizing/rescaling channels assuming these sigs
        sig2 = [0 2 4 8];
        iChan = [...
          2 3 4 ... % blur sig1(1:3)
          6:8 9:12 13:16 18:19 ... % SGS
          23:24 26:28 29:32 33:36]; % SLS
      end
      
      [S,SGS,SLS] = Features.pp(obj.I(iTrl),sig1,sig2,'hWaitBar',hWB);
      
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
    
    function cnts = channelDiagnostics(obj,iTrl,cnts)
      % Compute diagnostics on .I, .Ipp based in trials iTrl
      
      tfCntsSupplied = exist('cnts','var')>0;
      
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
    
    function [Is,nChan] = getCombinedIs(obj,iTrl)
      % Get .I combined with .Ipp for specified trials.
      %
      % iTrl: [nTrl] vector of trials
      %
      % Is: [nTrl] cell vec of image stacks [nr nc nChan] where
      %   nChan=1+numel(obj.Ipp)
      % nChan: number of TOTAL channels used/found
      
      nChanPP = numel(obj.IppInfo);
      fprintf(1,'Using %d additional channels.\n',nChanPP);
      
      nTrl = numel(iTrl);
      Is = cell(nTrl,1);
      for i=1:nTrl
        iT = iTrl(i);
        
        im = obj.I{iT};
        if nChanPP==0
          impp = nan(size(im,1),size(im,2),0);
        else
          impp = obj.Ipp{iT};
        end
        assert(size(impp,3)==nChanPP);
        Is{i} = cat(3,im,impp);
      end
      
      nChan = nChanPP+1;
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
    function [grps,ffd,ffdiTrl] = ffTrnSet(tblP,gvar)
      % Furthest-first training set analysis
      %
      % tblP: table with labeled positions (p)
      % gvar: field to use as grouping var
      %
      % grps: [Ngrp] categorical, unique groups found
      % ffd: [Ngrp] cell vec. ffd{i} contains a vector of "furthest-first"
      % distances, sorted in decreasing order.
      % ffdiTrl. [Ngrp] cell vec. ffdiTrl{i} is a vector of indices into 
      % tblP for ffd{i}.
            
      pTrn = tblP.p;
      g = tblP.(gvar);
      grps = categorical(unique(g));
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
        [~,~,tmpidx,~,mindists] = furthestfirst(pG,nG,'Start',[]);  
        warning(warnst);
        
        mindists(1) = inf;
        assert(isequal(sort(mindists,'descend'),mindists));
        
        ffd{iGrp} = mindists;
        ffdiTrl{iGrp} = iG(tmpidx);
      end
    end
    
    function hFig1 = ffTrnSetSelect(tblP,grps,ffd,ffdiTrl,varargin)
      % Display furthestfirst distances for groups in subplots; enable
      % clicking on subplots to visualize training shape
      
      fontsz = myparse(varargin,...
        'fontsize',8);
      
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
        bdfCbks{iGrp} = @(x,y)nst(x,y);
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
        
        % visualize
        warnst = warning('off','backtrace');
        [~,~,tmpidx,~,mindists] = furthestfirst(tblP.p(iTrnAcc,:),nTrnAcc,'Start',[]);
        warning(warnst);
        mindists(1) = inf;
        assert(isequal(sort(mindists,'descend'),mindists));
        
        figure;
        plot(mindists);
        grid('on');
        title('Total training furthestfirst dists','interpreter','none','fontweight','bold');
        ylabel('distance (px^2)');
        
        %           NSHOW = 6;
        %           iTrlShow1 = obj.iTrn(tmpidx(1:NSHOW));
        %           iTrlShow2 = obj.iTrn(tmpidx(end-NSHOW+1:end));
        %           obj.vizIdx(iTrlShow1);
        %           obj.vizIdx(iTrlShow2);
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