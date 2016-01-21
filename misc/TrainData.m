classdef TrainData < handle
  properties    
    Name    % Name of this TrainData
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
    function v = get.iUnused(obj)
      if ~isempty(intersect(obj.iTrn,obj.iTst))
        warning('TrainData:partition','Overlapping iTrn/iTst.');
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
    
    function obj = TrainData(Is,p,bb,md)
      assert(iscolumn(Is) && iscell(Is));
      N = numel(Is);
      assert(size(p,1)==N);
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
