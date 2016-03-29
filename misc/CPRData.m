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
  methods % property getters
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

      [iTrl,jan,romain] = myparse(varargin,...
        'iTrl',find(obj.isFullyLabeled),...
        'jan',false,...
        'romain',false);
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
          
%           nLblAct = nnz(tMD.tfAct(tfIDLbled));
%           nLblRst = nnz(tMD.tfRst(tfIDLbled));
          fprintf(1,'id %s. nfrm=%d. nfrmLbled=%d.\n',...
            id,nnz(tfID),nnz(tfIDLbled));
        end
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
      sMD = struct('iLbl',cell(0,1),'lblFile',[],'lblFileS',[],...
        'mov',[],'frm',[],'nLblInf',[],'nLblNaN',[],'nTag',[],'tagvec',[]);
      
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
          tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
          frmsLbled = find(tfLbled);
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
            if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
              im = rgb2gray(im);
            end
            
            fprintf('iLbl=%d, iMov=%d, read frame %d (%d/%d)\n',...
              iLbl,iMov,f,iFrm,nFrmRead);
            
            ITmp{iFrm} = im;
            lblsFrmXY = lpos(:,:,f);
            pTmp(iFrm,:) = Shape.xy2vec(lblsFrmXY);
            
            %tagbinvec = sum(tftagged(:,f)'.*2.^(0:npts-1));
            tagvec = find(tftagged(:,f)');
            
            sMD(end+1,1).iLbl = iLbl; %#ok<AGROW>
            sMD(end).lblFile = lblName;
            [~,sMD(end).lblFileS] = myfileparts(lblName);
            sMD(end).mov = movName;
            sMD(end).frm = f;
            sMD(end).nLblInf = sum(any(isinf(lblsFrmXY),2));
            sMD(end).nLblNaN = sum(any(isnan(lblsFrmXY),2));
            sMD(end).nTag = ntagged(f);
            sMD(end).tagvec = tagvec;
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
    
    function [iTrn,iTstAll,iTstLbl] = genITrnITst2(obj,idTest)
      % Given experiment id (testID), generate training and test exps.
            
      dfTest = idTest(1:9);
      
      tfLbled = obj.isFullyLabeled;
      obj.expandMDTable();
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
%       if ~ismember('actvf0',tMD.Properties.VariableNames)
%         tXLS = readtable(obj.MD_XLSFILE,'Sheet',obj.MD_XLSSHEET);
%         tf = ~cellfun(@isempty,tXLS.vid);
%         tXLS = tXLS(tf,:);
%         
%         tMD = join(tMD,tXLS); % should be joining on id)
%         
%         tMD.tfAct =  (tMD.frm>=tMD.actvf0 & tMD.frm<=tMD.actvf1);
%         tMD.tfRst = ~(tMD.frm>=tMD.actvf0 & tMD.frm<=tMD.actvf1);
%       end
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