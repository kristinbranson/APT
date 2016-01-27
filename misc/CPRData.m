classdef CPRData < handle
  properties    
    Name    % Name of this CPRData
    MD      % [NxR] Table of Metadata
    
    I       % [N] column cell vec, images
    pGT     % [NxD] GT shapes for I
    bboxes  % [Nx2d] bboxes for I 
    
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
      % obj = CPRData(Is,p,bb,md)
      % obj = CPRData(lblFiles,tfAllFrames)
      
      switch nargin
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
    
    function varargout = viz(obj,varargin)
      [varargout{1:nargout}] = Shape.viz(obj.I,obj.pGT,...
        struct('nfids',obj.nfids,'D',obj.D),'md',obj.MD,varargin{:});
    end
    
    function n = getFilename(obj)
      n = sprintf('td_%s_%s.mat',obj.Name,datestr(now,'yyyymmdd'));
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
    
    function H0 = histEq(obj,loadH0)
      if exist('loadH0','var')==0
        loadH0 = false;
      end
      
      imSz = cellfun(@size,obj.I,'uni',0);
      cellfun(@(x)assert(isequal(x,imSz{1})),imSz);
      imSz = imSz{1};
      imNpx = numel(obj.I{1});
        
      if loadH0
        assert(false,'dunno');
%         [fileH0,folderH0]=uigetfile('.mat');
%         load(fullfile(folderH0,fileH0));
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
        fprintf('using %d movies to form H0.\n',numel(idxuse));
        H0 = median(H(:,idxuse),2);
        H0 = H0/sum(H0)*imNpx(1);
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
    
  end
  
end
