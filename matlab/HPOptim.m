classdef HPOptim < handle
  
  % 1. Start with a split defn
  %    - outertrain vs outertest
  %    - outertrain has train/test xv splits
  % 2. Generate deltas/patches from current sPrm0.
  % 3. Fire xv jobs.
  % 4. When jobs all done. Compile results. Make plots. Select sPrmDelta.
  %   Update sPrm0. Goto 2. If sPrm0 is unchanged, stop. Take note of
  %   "downside" params that can be reduced without harm.
  % 5. Final train on full outertrain; apply to outertest.

  properties
    nview 
    
    basedir % fullpath 
    splitdirs % [nsplits] cellstr relative paths (relative to basedir)
    prmpat
    pchpat
    rndpat
    
    pchnames % [npch] cellstr 
    xverrsplits % [nsplitsx1] cell array. Each el is [nXVxnptsxnpchxnround]
    xvressplits % [nsplitsx1] cell array
    trntrkdxysplits % [nsplitsx1] cell array. Each el is [ntrk x nphyspt x (x/y) x nview x nround]
    
    prms % [nRound x 1]    
    tblres % [nRound x nSplits]
    tblrescomb % [nRound x 1]
  end
  properties (Dependent)
    nsplit
    nround
    npch
  end

  methods
    function v = get.nsplit(obj)
      v  = numel(obj.splitdirs);
    end
    function v = get.nround(obj)
      v = numel(obj.tblrescomb);
    end
    function v = get.npch(obj)
      v = numel(obj.pchnames);
    end
    function scores = getScores(obj)
      scores = nan(obj.npch,obj.nround,obj.nsplit);
      for iRnd=1:obj.nround
        for iSplt=1:obj.nsplit
          xverr = obj.xverrsplits{iSplt};
          tblRes = HPOptim.pchScores(xverr(:,:,:,iRnd),obj.pchnames);
          scores(:,iRnd,iSplt) = tblRes.score;
        end
      end
    end
    function pd = getPrmDir(obj,isplit,round)
      sdir = obj.splitdirs{isplit};
      prmdir = sprintf('prm%d',round-1);
      pd = fullfile(obj.basedir,sdir,prmdir);
    end
    function pd = getPchDir(obj,round)
      pchdir = sprintf(obj.pchpat,round-1);
      pd = fullfile(obj.basedir,pchdir);
    end
  end
  
  methods
    
    function obj = HPOptim(nview,varargin)
      % HPO workflow
      %
      % baseDir: dir containing all artifacts
      % splitDirs: [nsplit] cellstrs, relative paths (relative to baseDir)
      %   for results for various splits
      
      if nargin==0
        return;
      end
      
      [baseDir,prmPat,pchPat,rndPat,yearstr] = myparse(varargin,...
        'baseDir',pwd,...
        'prmPat','prm%d.mat',...
        'pchPat','pch%d',...
        'rndPat','prm%d',...
        'yearstr',datestr(now,'yyyy')...
        );      
      
      json = dir(fullfile(baseDir,'*.json'));
      if ~isscalar(json)
        error('Cannot find single json file to read.');
      end
      
      jsonS = json.name;
      json = fullfile(baseDir,jsonS);
      fprintf(1,'Found json %s in basedir %s.\n',jsonS,baseDir);
      
      json = readtxtfile(json);
      json = cat(2,json{:});
      json = jsondecode(json);
      splitinfo = struct2cell(json.splits);
      splitDirs = cellfun(@(x)x.dir,splitinfo,'uni',0);      
      
      nSplits = numel(splitDirs);
      
      % figure out nRounds
      iRound = 0;
      while 1
        prmFile = fullfile(baseDir,sprintf(prmPat,iRound));
        pchDir = fullfile(baseDir,sprintf(pchPat,iRound));
        if exist(prmFile,'file')==0
          fprintf(2,'Can''t find: %s\n',prmFile);
          break;
        end
        if exist(pchDir,'dir')==0
          fprintf(2,'Can''t find: %s\n',pchDir);
          break;
        end
        
        tfFoundAllRounds = true;
        for iSplit=1:nSplits
          rndDir = fullfile(baseDir,splitDirs{iSplit},sprintf(rndPat,iRound));
          if exist(rndDir,'dir')==0
            tfFoundAllRounds = false;
            break;
          end
          dd = dir(fullfile(rndDir,'*.mat'));
          fprintf(1,'... found %d mat-files in %s.\n',numel(dd),rndDir);
        end
        if ~tfFoundAllRounds
          fprintf(2,'Can''t find all split results for iRound %d.\n',iRound);
          break;
        end
        
        iRound = iRound+1;
      end
      
      nRound = iRound;
      if nRound==0
        error('No rounds/data found.');
      end
      fprintf(1,'Found %d rounds of HPO data.\n',nRound);
      
      prms = cell(nRound,1);
      xverrSplits = cell(nSplits,1);
      pchSplits = cell(nSplits,1);
      xvresSplits = cell(nSplits,1);
      %trntrkErrSplits = cell(nSplits,1); % [nxnptxnRound]
      trntrkDXYSplits = cell(nSplits,1); % [nxDxnRound]
      tftrntrkfound = true;
      for iRound=0:nRound-1
        pchDir = fullfile(baseDir,sprintf(pchPat,iRound));
        prmDirS = sprintf(prmPat,iRound);
        prmFileFull = fullfile(baseDir,prmDirS);
        [~,prmDirS,~] = fileparts(prmDirS);
        prms{iRound+1} = loadSingleVariableMatfile(prmFileFull);
        for iSplit=1:nSplits
          spltDir = fullfile(baseDir,splitDirs{iSplit});
          rndDir = fullfile(spltDir,sprintf(rndPat,iRound));
          xvresPat = sprintf('xv_*_%s_%%s_%s*.mat',prmDirS,yearstr);
          xvresBaseFile = sprintf('xv_*_%s_%s*.mat',prmDirS,yearstr);
          ddresBaseFile = dir(fullfile(rndDir,xvresBaseFile));
          assert(isscalar(ddresBaseFile),'Could not find base xv results: %s',xvresBaseFile);
          
          fprintf(1,'  Loading xv results from roundDir %s\n',rndDir);
          [xverrtmp,pchstmp,xvrestmp] = ...
            HPOptim.loadXVres(pchDir,rndDir,xvresPat,'xvbase',ddresBaseFile.name);
          if iRound==0
            xverrSplits{iSplit} = xverrtmp;
            pchSplits{iSplit} = pchstmp;
            xvresSplits{iSplit} = xvrestmp;
          else
            xverrSplits{iSplit}(:,:,:,end+1) = xverrtmp;
            pchSplits{iSplit}(:,end+1) = pchstmp;
            xvresSplits{iSplit}(:,end+1) = xvrestmp;
          end
          
          trntrkPat = sprintf('trntrk_*_%s_%s*.mat',prmDirS,yearstr);
          trntrkFile = dir(fullfile(spltDir,trntrkPat));
          switch numel(trntrkFile)
            case 0
              fprintf(2,'  Did not find trntrk file for (split,iround)=(%d,%d).\n',iSplit,iRound);
              %errLblTrk = nan; % will lead to harderr if iRound==0; otherwise will sing expand
              dLblTrk = nan; % etc
              if iRound==0
                tftrntrkfound = false;
              end
            case 1
              fprintf(1,'  Found trntrk data for (split,iround)=(%d,%d).\n',iSplit,iRound);
              tt = load(fullfile(spltDir,trntrkFile.name),'-mat');
              errLblTrk = tt.tblRes.dLblTrk;
              dLblTrk = tt.tblRes.pTrk-tt.tblRes.pLbl;
              dLblTrktmp = reshape(dLblTrk,size(dLblTrk,1),size(dLblTrk,2)/2,[]);
              dCheckE = sqrt(squeeze(sum(dLblTrktmp.^2,3))) - errLblTrk;
              fprintf(1,'  ..Sanity checks dxy: max(abs(resid)))=%.3g\n',...
                max(abs(dCheckE(:))));
            otherwise
              assert(false);
          end
          if iRound==0
            %trntrkErrSplits{iSplit} = errLblTrk; % [nxnpt]
            trntrkDXYSplits{iSplit} = dLblTrk; % [nxD]
          else
            %trntrkErrSplits{iSplit}(:,:,end+1) = errLblTrk;
            trntrkDXYSplits{iSplit}(:,:,end+1) = dLblTrk;
          end
        end
      end
      
      for iSplit=1:nSplits
        [n,npts,npch,nroundtmp] = size(xverrSplits{iSplit});
        assert(nroundtmp==nRound);
        fprintf(1,'Split type %d. (n,npts,npch) = (%d,%d,%d).\n',iSplit,n,npts,npch);
        
        if tftrntrkfound
          dxy = trntrkDXYSplits{iSplit};
          [ntrk,D,nRound2] = size(dxy);
          assert(nRound2==nRound);
          nPhysPts = D/nview/2;
          dxy = reshape(dxy,ntrk,nPhysPts,nview,2,nRound); % n, pt, view, x/y, set
          dxy = permute(dxy,[1 2 4 3 5]); % n, pt, x/y, view, set
          trntrkDXYSplits{iSplit} = dxy;
        end
      end
      assert(isequal(pchSplits{:},repmat(pchSplits{1}(:,1),1,nRound)));
      pchs = pchSplits{1}(:,1);
      
      %% scores by round
      tblRes = cell(nRound,nSplits);
      tblResComb = cell(nRound,1);
      %scores = nan(nPch,nRounds,2); % pch, round, easy/hard
      for iRnd=1:nRound
        for iSplit=1:nSplits
          t = HPOptim.pchScores(xverrSplits{iSplit}(:,:,:,iRnd),pchs);
          [~,t.scrrank] = sort(t.score,'descend');
          
          tblRes{iRnd,iSplit} = t;
          tblfldsassert(t,{'score' 'nptimprove' 'nptimprovedfull' 'pch' 'scrrank'});
          t = t(:,[1 5 3 4]);
          t.Properties.VariableNames([1 3]) = {'scr' 'nptimp'};
          t.Properties.VariableNames([1 2 3]) = ...
            cellfun(@(x)sprintf('%s_splt%d',x,iSplit),t.Properties.VariableNames([1 2 3]),'uni',0);
          if iSplit==1
            tblResComb{iRnd} = t;
          else
            tblResComb{iRnd} = innerjoin(tblResComb{iRnd},t,'Keys',{'pch'});
          end
        end
        
        colnames = tblResComb{iRnd}.Properties.VariableNames;
        scrcols = startsWith(colnames,'scr_');
        scrrankcols = startsWith(colnames,'scrrank');
        nptimpcols = startsWith(colnames,'nptimp_');
        assert(isequal(nSplits,nnz(scrcols),nnz(scrrankcols),nnz(nptimpcols)));
        tblResComb{iRnd}.scrtot = sum(tblResComb{iRnd}{:,scrcols},2);
        tblResComb{iRnd}.scrranktot = sum(tblResComb{iRnd}{:,scrrankcols},2);
        tblResComb{iRnd}.nptimptot = sum(tblResComb{iRnd}{:,nptimpcols},2);
        [~,idx] = sort(tblResComb{iRnd}.scrranktot,'ascend');
        tblResComb{iRnd} = tblResComb{iRnd}(idx,:);
        
        [~,cols] = ismember({'pch' 'scrtot' 'scrranktot' 'nptimptot'},...
          tblflds(tblResComb{iRnd}));
        ncols = size(tblResComb{iRnd},2);
        cols = [cols setdiff(1:ncols,cols)]; %#ok<AGROW>
        tblResComb{iRnd} = tblResComb{iRnd}(:,cols);
      end
      
      obj.nview = nview;
      
      obj.basedir = baseDir;
      obj.splitdirs = splitDirs;
      obj.prmpat = prmPat;
      obj.pchpat = pchPat;
      obj.rndpat = rndPat;
      
      obj.pchnames = pchs;
      obj.xverrsplits = xverrSplits;
      obj.xvressplits = xvresSplits;
      obj.trntrkdxysplits = trntrkDXYSplits;
      
      obj.prms = prms;
      obj.tblres = tblRes;
      obj.tblrescomb = tblResComb;
    end
    
    function acceptPchs(obj,ipchs)
      % ipchs: vector of rows of .tblrescomb to accept (1-based row indices)
      
      iroundbase = obj.nround-1;
      iroundnew = obj.nround;
      basePrmFile = fullfile(obj.basedir,sprintf(obj.prmpat,iroundbase));
      newPrmFile = fullfile(obj.basedir,sprintf(obj.prmpat,iroundnew));
      basePchDir = fullfile(obj.basedir,sprintf(obj.pchpat,iroundbase));
      newPchDir = fullfile(obj.basedir,sprintf(obj.pchpat,iroundnew));
      assert(exist(basePrmFile,'file')>0);
      assert(exist(newPrmFile,'file')==0);
      assert(exist(basePchDir,'dir')>0);
      assert(exist(newPchDir,'dir')==0);
      
      pchSel = obj.tblrescomb{obj.nround}.pch(ipchs);
      obj.genNewPrmFile(basePrmFile,newPrmFile,basePchDir,pchSel);
      
      obj.genAndWritePchs(newPrmFile,newPchDir,{});
    end
    
    function hFig = plotConvergence(obj,varargin)
      [dosave,savedir] = myparse(varargin,...
        'dosave',false,...
        'savedir','figs'...
        );
      
      hFig = [];
      
      JITDX = 0.2;
      CLRS = {[0 0 1] [0 0.75 0]}; % easy/hard
      MRKRSZ = 30;
      
      hFig(end+1) = figure(31);
      hfig = hFig(end);
      clf(hfig);
      set(hfig,'Name','Convergence','Position',[1 41 1920 963]);
      ax = axes;
      hold(ax,'on');
      grid(ax,'on');
      nRounds = obj.nround;
      nSplits = obj.nsplit;
      nPch = obj.npch;
      scores = obj.getScores();
      for iRnd=1:nRounds
        for iSplit=1:nSplits
          x = (nRounds+1)*(iSplit-1) + repmat(iRnd,nPch,1);
          x = x + 2*JITDX*(rand(nPch,1)-0.5);
          y = scores(:,iRnd,iSplit);
          plot(ax,x,y,'.','markersize',MRKRSZ,'color',CLRS{iSplit});
        end
      end
      xlim(ax,[-0.5 nSplits*nRounds+2]);
      %ylim(ax,[-20 10]);
      %       set(ax,'XTick',[1:4 6:9],'XTickLabel',...
      %         {'Rnd1/easy' 'Rnd2/easy' 'Rnd3/easy' 'Rnd4/easy' 'Rnd1/hard' 'Rnd2/hard' 'Rnd3/hard' 'Rnd4/hard'},...
      %         'XTickLabelRotation',0,'fontsize',18);
      ylabel(ax,'XV prctile improvement score');
      set(ax,'fontweight','bold');
      title(ax,'HPO Convergence (?)');
      
      if dosave
        for i=1:numel(hFig)
          h = figure(hFig(i));
          fname = h.Name;
          hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
          set(h,'PaperOrientation','landscape','PaperType','arch-d');
          print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));
          print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));
          fprintf(1,'Saved %s.\n',fname);
        end
      end
      
    end
    
    function hFig = plotNoPatchXVerr(obj,varargin)
      [iNoPatch,ptiles,ptilesbig,iptsplot,ignorerows] = myparse(varargin,...
        'iNoPatch',1,...
        'ptiles',70:3:99,...
        'ptilesbig',98:.5:99.5,...
        'iptsplot',[],...% PHYSICAL pts to plot 9 11 12:17]... % bub
        'ignorerows',[]... % (opt) [nsplit] cell, ignorerows{isplit} contains vector of row indices into xverrsplits{isplit} to REMOVE
      );
      
      assert(strcmp(obj.pchnames{iNoPatch},'NOPATCH'));
      
      if strcmp(iptsplot,'bub')
        iptsplot = [9 11 12:17];
      end
      
      hFig = [];
      
      for isplit=1:obj.nsplit
        xverr = obj.xverrsplits{isplit};
        if ~isempty(ignorerows) && ~isempty(ignorerows{isplit})
          xverr(ignorerows{isplit},:,:,:) = [];
        end          
        
        [nxv,nptstot,npch,nround] = size(xverr);
        nphyspts = nptstot/obj.nview;
        if isempty(iptsplot)
          iptsplot = 1:nphyspts;
        end
        nptsPlot = numel(iptsplot);
        xverr = reshape(xverr,nxv,nphyspts,obj.nview,npch,nround);
        xverrNP = xverr(:,iptsplot,:,iNoPatch,:); % nxv,nptsplot,nvw,1,nround
        xverrNP = reshape(xverrNP,nxv,nptsPlot,1,obj.nview,obj.nround);
        
        fignum = isplit*10 + 1;
        hFig(end+1,1) = figure(fignum);
        hfig = hFig(end);
        title = sprintf('NoPatch Split%d',isplit);
        set(hfig,'Position',[1 41 1920 963],'name',title);
        GTPlot.ptileCurves(xverrNP,...
          'ptiles',ptiles,......
          'hFig',hfig,...
          'axisArgs',{'XTicklabelRotation',45,'FontSize' 16}...
          );
        
        fignum = isplit*10 + 2;
        hFig(end+1,1) = figure(fignum);
        hfig = hFig(end);
        title = sprintf('NoPatch Big Ptile Split%d',isplit);
        set(hfig,'Position',[1 41 1920 963],'name',title);
        GTPlot.ptileCurves(xverrNP,...
          'ptiles',ptilesbig,......
          'hFig',hfig,...
          'axisArgs',{'XTicklabelRotation',45,'FontSize' 16}...
          );
      end
    end
    
    function hFig = plotTrnTrkErr(obj,varargin)
      [dosave,savedir,ptiles,ptilesbig,figpos,iptsplot,IBE,pLblBE] = myparse(varargin,...
        'dosave',false,...
        'savedir','figs',...
        'ptiles',[50 75 90 95],...
        'ptilesbig',97.5:.5:99,...
        'figpos',[1 1 1920 960],...
        'iptsplot',[],... [9 11 12:17],... % bub
        'IBE',[],... % (opt) if supplied, [nview] (cropped) ims for use with bullseye plots
        'pLblBE',[]... % (opt) [nphyspt x 2 x nview]
        );
      
      tfBE = ~isempty(IBE);
              
      setNames = arrayfun(@(x)sprintf('Round%d',x),0:obj.nround-1,'uni',0);
      
      hFig = [];
      
      for isplit=1:obj.nsplit
        dxySplit = obj.trntrkdxysplits{isplit}; % [ntrk x nphyspt x (x/y) x nview x nround]
        if isempty(iptsplot)
          iptsplot = 1:size(dxySplit,2);
        elseif strcmp(iptsplot,'bub')
          iptsplot = [9 11 12:17];
        end
        ptnames = arrayfun(@(x)sprintf('pt%d',x),iptsplot,'uni',0);
        dxySplit = dxySplit(:,iptsplot,:,:,:);
        
        tstr = sprintf('split%d',isplit);
        fignum = 10*isplit+1;
        hFig(end+1) = figure(fignum);
        hfig = hFig(end);
        set(hfig,'Name',tstr,'Position',figpos);
        [~,ax] = GTPlot.ptileCurves(...
          dxySplit,'hFig',hfig,...
          'setNames',setNames,...
          'ptnames',ptnames,...
          'ptiles',ptiles,...
          'axisArgs',{'XTicklabelRotation',90,'FontSize',16});

        tstr = sprintf('split%d bigPtiles',isplit);
        fignum = 10*isplit+2;
        hFig(end+1) = figure(fignum);
        hfig = hFig(end);
        set(hfig,'Name',tstr,'Position',figpos);
        [~,ax] = GTPlot.ptileCurves(...
          dxySplit,'hFig',hfig,...
          'setNames',setNames,...
          'ptnames',ptnames,...
          'ptiles',ptilesbig,...
          'axisArgs',{'XTicklabelRotation',90,'FontSize',16});
        
        if tfBE
          tstr = sprintf('split%d BE',isplit);
          fignum = 10*isplit+2;
          hFig(end+1) = figure(fignum);
          hfig = hFig(end);
          set(hfig,'Name',tstr,'Position',figpos);
          [~,ax] = GTPlot.bullseyePtiles(dxySplit,...
            IBE,pLblBE(iptsplot,:,:),...
            'hFig',hfig,...
            'setNames',setNames,...
            'ptiles',ptiles,...
            'lineWidth',2,...
            'axisArgs',{'XTicklabelRotation',90,'FontSize',16});
          
          tstr = sprintf('split%d BEell',isplit);
          fignum = 10*isplit+3;
          hFig(end+1) = figure(fignum);
          hfig = hFig(end);
          set(hfig,'Name',tstr,'Position',figpos);
          [~,ax] = GTPlot.bullseyePtiles(dxySplit,...
            IBE,pLblBE(iptsplot,:,:),...
            'hFig',hfig,...
            'setNames',setNames,...
            'ptiles',ptiles,...
            'lineWidth',1,... %  'axisArgs',{'XTicklabelRotation',90,'FontSize',16},...
            'contourtype','ellipse');
        end
      end

      if dosave
        for i=1:numel(hFig)
          h = figure(hFig(i));
          fname = h.Name;
          hgsave(h,fullfile(savedir,[fname '.fig']));
          set(h,'PaperOrientation','landscape','PaperType','arch-d');
          print(h,'-dpdf',fullfile(savedir,[fname '.pdf']));
          print(h,'-dpng','-r300',fullfile(savedir,[fname '.png']));
          fprintf(1,'Saved %s.\n',fname);
        end
      end
      
    end
    
    function [hFig,scoreSR] = plotTrnTrkErrWithSelect(obj,varargin)
      [dosave,savedir,ptiles,figpos,fignums,iptsplot] = myparse(varargin,...
        'dosave',false,...
        'savedir','figs',...
        'ptiles',[50 90],... % ptiles to include in i) plot and ii) overall score leading to selection
        'figpos',[1 1 1920 960],...
        'fignums',[11 12],...
        'iptsplot',[]... % point indices to include in i) plots and ii) overall score leading to selection
          ... %         'IBE',[],... % (opt) if supplied, [nview] (cropped) ims for use with bullseye plots
          ... %         'pLblBE',[]... % (opt) [nphyspt x 2 x nview]
        );
      
      %tfBE = ~isempty(IBE);
                          
      dxyAllSplits = cell(obj.nsplit,1);
      for isplit=1:obj.nsplit
        dxySplit = obj.trntrkdxysplits{isplit}; % [ntrk x nphyspt x (x/y) x nview x nround]
        if isempty(iptsplot)
          iptsplot = 1:size(dxySplit,2);
        elseif strcmp(iptsplot,'bub')
          iptsplot = [9 11 12:17];
        end
        dxySplit = dxySplit(:,iptsplot,:,:,:);
        [ntrk,nptsplot,d,nvw,nrnd] = size(dxySplit);
        assert(d==2);
        assert(nrnd==obj.nround);       
        dxySplit = permute(dxySplit,[1 2 4 3 5]);
        dxySplit = reshape(dxySplit,[ntrk nptsplot*nvw d 1 nrnd]); % collapse all views. actually i think originally nvw==1 already with all pts collapsed 
        
        dxyAllSplits{isplit} = dxySplit; % splits will become "views" in ptileplot
      end            
    
      ptnames = arrayfun(@(x)sprintf('pt%d',x),iptsplot,'uni',0);
      viewNames = arrayfun(@(x)sprintf('splt%d',x),1:obj.nsplit,'uni',0);
      setNames = arrayfun(@(x)sprintf('Rnd%d',x),0:obj.nround-1,'uni',0);
        
      [hFig,scoreSR] = GTPlot.ptileCurvesPickBest(dxyAllSplits,ptiles,...
        'viewNames',viewNames,...
        'setNames',setNames,...
        'errCellPerView',true,...
        'fignums',fignums,...
        'figpos',figpos,...
        'ptileCurvesArgs',{'ptnames' ptnames});
            
      %tstr = 'trntrkerrWithSelect';      
      
%         if tfBE
%           tstr = sprintf('split%d BE',isplit);
%           fignum = 10*isplit+2;
%           hFig(end+1) = figure(fignum);
%           hfig = hFig(end);
%           set(hfig,'Name',tstr,'Position',figpos);
%           [~,ax] = GTPlot.bullseyePtiles(dxySplit,...
%             IBE,pLblBE(iptsplot,:,:),...
%             'hFig',hfig,...
%             'setNames',setNames,...
%             'ptiles',ptiles,...
%             'lineWidth',2,...
%             'axisArgs',{'XTicklabelRotation',90,'FontSize',16});
%           
%           tstr = sprintf('split%d BEell',isplit);
%           fignum = 10*isplit+3;
%           hFig(end+1) = figure(fignum);
%           hfig = hFig(end);
%           set(hfig,'Name',tstr,'Position',figpos);
%           [~,ax] = GTPlot.bullseyePtiles(dxySplit,...
%             IBE,pLblBE(iptsplot,:,:),...
%             'hFig',hfig,...
%             'setNames',setNames,...
%             'ptiles',ptiles,...
%             'lineWidth',1,... %  'axisArgs',{'XTicklabelRotation',90,'FontSize',16},...
%             'contourtype','ellipse');
%         end

      if dosave
        for i=1:numel(hFig)
          h = figure(hFig(i));
          fname = h.Name;
          hgsave(h,fullfile(savedir,[fname '.fig']));
          set(h,'PaperOrientation','landscape','PaperType','arch-d');
          print(h,'-dpdf',fullfile(savedir,[fname '.pdf']));
          print(h,'-dpng','-r300',fullfile(savedir,[fname '.png']));
          fprintf(1,'Saved %s.\n',fname);
        end
      end
      
    end
    
    
    function showPrmDiff(obj)
      pcell = obj.prms;
      
      nrnd = numel(pcell);
      for irnd=2:nrnd
        fprintf(1,'\n### Round %d->%d\n',irnd-1,irnd);
        structdiff(pcell{irnd-1},pcell{irnd});
      end
      
      if nrnd>1
        fprintf(1,'\n### NET Round %d->%d\n',1,nrnd);
        structdiff(pcell{1},pcell{nrnd});
      end
    end
    
    function s = getPerfDataAll(obj)
      for isplit=1:obj.nsplit
      for iround=1:obj.nround
        s(iround,isplit) = obj.getPerfData(isplit,iround);
        fprintf('split %d round %d\n',isplit,iround);
      end
      end
    end
    function s = getPerfData(obj,isplit,iround)
      %iround: 1-based
      
      pnames = obj.pchnames;            
      prmdir = obj.getPrmDir(isplit,iround);
      pchdir = obj.getPchDir(iround);
      s = struct();
      prm0 = obj.prms{iround};
      
      for pch=pnames(:)',pch=pch{1};
        dirpat = fullfile(prmdir,sprintf('apt-20*-%s.log',pch));
        dd = dir(dirpat);
        if isscalar(dd)
          logf = fullfile(prmdir,dd.name);
          elapsedcmd = sprintf('grep -A 1 Elapsed %s',logf);
          [st,res] = system(elapsedcmd);
          res = regexp(res,'\n','split');
          elapsedpat = 'Elapsed time is (?<sec>[0-9\.]+) seconds.';                
          try
            switch obj.nview
              case 1
                assert(all(strncmp(res([1 4 7]),'Elapsed',numel('Elapsed'))));
                assert(all(strncmp(res([2 5 8]),'Tracking',numel('Tracking'))));
                secs = [];
                for row=[1 4 7]
                  toks = regexp(res{row},elapsedpat,'tokens');
                  assert(isscalar(toks) && isscalar(toks{1}));
                  secs(1,end+1) = round(str2double(toks{1}{1}));
                end
              case 2
                assert(all(strncmp(res(1:3:end),'Elapsed',numel('Elapsed'))));
                assert(all(strncmp(res(5:6:end),'Tracking',numel('Tracking'))));
                secs = nan(2,3);
                for ifold=1:3
                for iview=1:2
                  row = 1 + (iview-1)*3 + (ifold-1)*6;
                  toks = regexp(res{row},elapsedpat,'tokens');
                  assert(isscalar(toks) && isscalar(toks{1}));
                  secs(iview,ifold) = round(str2double(toks{1}{1}));
                end
                end
              otherwise
                assert(false,'Unsupported number of views.');
            end
          catch ME
            warningNoTrace('Unexpected parse of %s.\n',logf);
            secs = nan(obj.nview,3);
          end
        else
          warningNoTrace('No log found for %s.',dirpat);
          secs = nan(obj.nview,3);
        end
                  
        szassert(secs,[obj.nview 3]);
        s.(pch).tracktimes = secs;          

        dirpat = fullfile(pchdir,sprintf('%s.m',pch));
        dd = dir(dirpat);
        if isscalar(dd)
          pchf = fullfile(pchdir,dd.name);
          pchcmd = sprintf('cat %s',pchf);
          [st,res] = system(pchcmd);
          res = regexp(res,'\n','split');
          res = res{1};
          pchpat = '(?<lhs>[^=]+)=(?<rhs>.+)';
          toks = regexp(res,pchpat,'names');
          
          evalstr = ['prm0' toks.lhs];
          prmb4 = eval(evalstr);
          prmaf = str2double(toks.rhs);
          s.(pch).b4aft = [prmb4 prmaf];
          s.(pch).b4aftstr = sprintf('%.2f -> %.2f',prmb4,prmaf);
        else
          warningNoTrace('No patch found for %s.',dirpat);
          s.(pch).b4aft = [nan nan];
          s.(pch).b4aft = 'unk -> unk';
        end
        
        tblres = obj.tblres{iround,isplit};
        idx = find(strcmp(pch,tblres.pch));
        assert(isscalar(idx));
        sres = table2struct(tblres(idx,:));
        sres = rmfield(sres,'pch');
        s.(pch) = structmerge(s.(pch),sres);
      end
    end
        
  end
  
  methods (Static)
    
    function tbl = perfSummFromPerfData(sPD,pchs)
      % sPD: output of getPerfData
      % pchs: eg {'NrepTrk_up' 'NrepTrk_dn' ... };
      
      if exist('pchs','var')==0
        pchs = {'NrepTrk_up' 'NrepTrk_dn' 'NumMajorIter_up' 'NumMajorIter_dn' 'NumMinorIter_up' 'NumMinorIter_dn'};
      end
      
      s = [];
      
      [nround,nsplit] = size(sPD);
      for pch = pchs(:)',pch=pch{1};
      for iround=1:nround
      for isplit=1:nsplit
        s0 = sPD(iround,isplit).NOPATCH;
        s1 = sPD(iround,isplit).(pch);
        trktimesrat = s1.tracktimes./s0.tracktimes;
        trktimesratmn = nanmean(trktimesrat(:));
        
        s(end+1,1).pch = pch;
        s(end).round = iround;
        s(end).split = isplit;
        szassert(s1.b4aft,[1 2]);
        s(end).b4 = s1.b4aft(1);
        s(end).b4aftstr = s1.b4aftstr;
        s(end).trktimesbase = s0.tracktimes;
        s(end).trktimesrat = trktimesrat;
        s(end).trktimesratmn = trktimesratmn;
        s(end).score = s1.score;
        s(end).nptimpfull = s1.nptimprovedfull;        
      end
      end
      end
      
      tbl = struct2table(s);      
    end
    
    function tres = perfSummSummRowFcn(tgrp)
      trkTratmnmn = round((nanmean(tgrp.trktimesratmn)-1)*100);
      %trkTratmnmdn = median(tgrp.trktimesratmn);
      trkTratmnstd = round(nanstd(tgrp.trktimesratmn)*100);
      scrmn = nanmean(tgrp.score);
      %scrmdn = median(tgrp.score);
      scrstd = nanstd(tgrp.score);
      nptimpfullmn = nanmean(tgrp.nptimpfull);
      %nptimpfullmdn = median(tgrp.nptimpfull);
      nptimpfullstd = nanstd(tgrp.nptimpfull);
      assert(all(strcmp(tgrp.pch,tgrp.pch{1})));
      assert(all(strcmp(tgrp.b4aftstr,tgrp.b4aftstr{1})));
      pch = tgrp.pch(1); % want cell for table() single-row API quirk
      b4aftstr = tgrp.b4aftstr(1); % etc
      count = height(tgrp);
      b4 = tgrp.b4(1);
      tres = table(pch,b4,b4aftstr,count,trkTratmnmn,trkTratmnstd,...
        scrmn,scrstd,...
        nptimpfullmn,nptimpfullstd);
    end
    function tbl = perfSummSumm(tblPerfSumm)
      t = tblPerfSumm;
      keys = strcat(t.pch,'#',t.b4aftstr);
      keyun = unique(keys);
      tbl = [];
      for k=keyun(:)',k=k{1};
        tf = strcmp(k,keys);
        tgrp = t(tf,:);
        tresgrp = HPOptim.perfSummSummRowFcn(tgrp);
        tbl = cat(1,tbl,tresgrp);
      end
      
      tbl = sortrows(tbl,{'pch' 'b4'});
      
        
%       tbl = rowfun(@HPOptim.perfSummSummRowFcn,tblPerfSumm,...
%         'InputVariables',{'pch' 'b4aftstr' 'trktimesratmn' 'score' 'nptimpfull'},...
%         'GroupingVariables',{'pch' 'b4aftstr'},...
%         'OutputVariableNames',{'pch' 'b4aftstr' 'trkTratmnmn' 'trkTratmnmdn' 'scrmn' 'scrmdn' 'nptimfullmn' 'nptimfullmdn'},...
%         'NumOutputs',8, ...
%         'OutputFormat','table'...
%         );
    end
    
    function [xverr,pchNames,xvres] = loadXVres(pchDir,xvDir,xvPat,varargin)
      % xvPat: sprintf-pat given a pchname to form xv-results-filename
      %
      % xverr: [nXVxnptsxnpch] (npts==nPhysPtsxnView)
      % pchNames: [npch] cellstr
      
      xvbase = myparse(varargin,...
        'xvbase',''); % if supplied, load 'base' xv results with this name
      tfBase = ~isempty(xvbase);
      
      % Get patches
      dd = dir(fullfile(pchDir,'*.m'));
      pchs = {dd.name}';
      if tfBase
        pchs = [{'DUMMY_UNUSED'}; pchs];
      end
      npch = numel(pchs);
      
      %% Load res
      xvres = cell(npch,1);
      pchNames = cell(npch,1);
      for i=1:npch
        if tfBase && i==1
          xvresfnameS = xvbase;
          pchS = 'NOPATCH';
        else
          [~,pchS,~] = fileparts(pchs{i});
          pat = sprintf(xvPat,pchS);
          dd = dir(fullfile(xvDir,pat));
          if isempty(dd)
            warningNoTrace('No xv results found for pat: %s\n',pat);
            xvresfnameS = '';
          elseif isscalar(dd)
            xvresfnameS = dd.name;
          else
            assert(false);
          end
        end
        if ~isempty(xvresfnameS)
          xvresfname = fullfile(xvDir,xvresfnameS);
          assert(exist(xvresfname,'file')>0);
          xvres{i,1} = load(xvresfname);
        end
        pchNames{i} = pchS;
        fprintf(1,'%d ',i);
      end
      fprintf('\n');
      %       xvres0 = {...
      %         load(fullfile(XVRESDIR,'xv_tMain4523_tMain4523_split3_easy_prmMain0_20180710T103811.mat')) ...
      %         load(fullfile(XVRESDIR,'xv_tMain4523_tMain4523_split3_hard_prmMain0_20180710T104734.mat'))};
      %       xvres = [xvres0; xvres];
      %       pchNames = [{'base'}; pchNames];
      
      %%
      %       assert(npch+1==size(xvres,1));
      nXV = height(xvres{1}.xvRes); % hopefully xvres{1} existed/got loaded
      npts = size(xvres{1}.xvRes.dGTTrk,2);
      fprintf(1,'nXV=%d, npts=%d, %d patches.\n',nXV,npts,npch);
      xverr = nan(nXV,npts,npch);
      for i=1:npch
        if ~isempty(xvres{i})
          xverr(:,:,i) = xvres{i}.xvRes.dGTTrk;
        end
      end
    end
    
    function aptClusterCmd(roundid,tblfile,spltfile,prmfile,pchdir,varargin)
      bindate = myparse(varargin,...
        'bindate','20180713.feature.deeptrack');
      
      blapt = '/groups/branson/bransonlab/apt';
      hpo = fullfile(blapt,'tmp','hpo');
      cmd = sprintf('/groups/branson/home/leea30/git/aptdl/APTCluster.py -n 6 --outdir %s/%s --bindate %s --trackargs "tableFile %s/%s tableSplitFile %s/%s paramFile %s/%s" --prmpatchdir %s/%s /groups/branson/bransonlab/apt/experiments/data/sh_trn4523_gt080618_made20180627_cacheddata.lbl xv',...
        hpo,roundid,bindate,hpo,tblfile,hpo,spltfile,hpo,prmfile,hpo,pchdir);
      fprintf(1,'Run:\n%s\n',cmd);
    end
    
    function genAndWritePchs(basePrmFile,pchDir,genPchsArgs)
      sPrm0 = loadSingleVariableMatfile(basePrmFile);
      s = HPOptim.genPchs(sPrm0,genPchsArgs{:});
      HPOptim.writePchDir(pchDir,s);
      fprintf(1,'Done creating pchdir: %s\n',pchDir);
    end
    
    function s = genPchs(sPrm0,varargin)
      [iterFac,radFac,midFac] = myparse(varargin,...
        'iterFac',1.4,...
        'radFac',1.6,...
        'midFac',1.5...
        );
      
      s = struct();
      
      he = sPrm0.ROOT.ImageProcessing.HistEq.Use;      
      s.HistEq_flip = {sprintf('.ROOT.ImageProcessing.HistEq.Use=%d',~he)};
      
      T0 = sPrm0.ROOT.CPR.NumMajorIter;
      s.NumMajorIter_up = {sprintf('.ROOT.CPR.NumMajorIter=%d',round(iterFac*T0))};
      s.NumMajorIter_dn = {sprintf('.ROOT.CPR.NumMajorIter=%d',round(1/iterFac*T0))};
      
      K = sPrm0.ROOT.CPR.NumMinorIter;
      s.NumMinorIter_up = {sprintf('.ROOT.CPR.NumMinorIter=%d',round(iterFac*K))};
      s.NumMinorIter_dn = {sprintf('.ROOT.CPR.NumMinorIter=%d',round(1/iterFac*K))};
      
      FD = sPrm0.ROOT.CPR.Ferns.Depth;
      s.FernsDepth_up = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD+1)};
      s.FernsDepth_up2 = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD+2)};
      s.FernsDepth_dn = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD-1)};
      
      FTlo = sPrm0.ROOT.CPR.Ferns.Threshold.Lo;
      FThi = sPrm0.ROOT.CPR.Ferns.Threshold.Hi;
      FTmid = (FTlo+FThi)/2;
      FTrad = (FThi-FTlo)/2;
      s.FernThresholdMid_up = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+2*FTrad); };
      s.FernThresholdMid_dn = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-2*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid); };
      
      s.FernThresholdRad_up = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-2*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+2*FTrad); };
      s.FernThresholdRad_dn = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-0.5*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+0.5*FTrad); };
      
      RF0 = sPrm0.ROOT.CPR.Ferns.RegFactor;
      s.RegFactor_up = { sprintf('.ROOT.CPR.Ferns.RegFactor=%.3f',2*RF0) };
      s.RegFactor_dn = { sprintf('.ROOT.CPR.Ferns.RegFactor=%.3f',0.5*RF0) };
      
      TwoLMRad0 = sPrm0.ROOT.CPR.Feature.Radius;
      s.TwoLMRad_up = { sprintf('.ROOT.CPR.Feature.Radius=%.3f',radFac*TwoLMRad0) };
      s.TwoLMRad_dn = { sprintf('.ROOT.CPR.Feature.Radius=%.3f',1/radFac*TwoLMRad0) };
      
      TwoLMABRat0 = sPrm0.ROOT.CPR.Feature.ABRatio;
      s.TwoLMABRat_up = { sprintf('.ROOT.CPR.Feature.ABRatio=%.3f',radFac*TwoLMABRat0) };
      s.TwoLMABRat_dn = { sprintf('.ROOT.CPR.Feature.ABRatio=%.3f',1/radFac*TwoLMABRat0) };
      
      s.OneLM_lo = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=30' };
      s.OneLM_md = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=60' };
      s.OneLM_hi = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=90' };
      
      FtrNGen0 = sPrm0.ROOT.CPR.Feature.NGenerate;
      s.FtrNGen_up = { sprintf('.ROOT.CPR.Feature.NGenerate=%d',round(radFac*FtrNGen0)) };
      s.FtrNGen_dn = { sprintf('.ROOT.CPR.Feature.NGenerate=%d',round(1/radFac*FtrNGen0)) };
      
      FtrNstd0 = sPrm0.ROOT.CPR.Feature.Nsample_std;
      s.FtrNstd_up = { sprintf('.ROOT.CPR.Feature.Nsample_std=%d',round(radFac*FtrNstd0)) };
      s.FtrNstd_dn = { sprintf('.ROOT.CPR.Feature.Nsample_std=%d',round(1/radFac*FtrNstd0)) };
      
      FtrNcor0 = sPrm0.ROOT.CPR.Feature.Nsample_cor;
      s.FtrNcor0_up = { sprintf('.ROOT.CPR.Feature.Nsample_cor=%d',round(midFac*FtrNcor0)) };
      s.FtrNcor0_dn = { sprintf('.ROOT.CPR.Feature.Nsample_cor=%d',round(1/midFac*FtrNcor0)) };
      
      NrepTrn0 = sPrm0.ROOT.CPR.Replicates.NrepTrain;
      s.NrepTrn_up = { sprintf('.ROOT.CPR.Replicates.NrepTrain=%d',round(iterFac*NrepTrn0)) };
      s.NrepTrn_dn = { sprintf('.ROOT.CPR.Replicates.NrepTrain=%d',round(1/iterFac*NrepTrn0)) };
      
      NrepTrk0 = sPrm0.ROOT.CPR.Replicates.NrepTrack;
      s.NrepTrk_up = { sprintf('.ROOT.CPR.Replicates.NrepTrack=%d',round(iterFac*NrepTrk0)) };
      s.NrepTrk_dn = { sprintf('.ROOT.CPR.Replicates.NrepTrack=%d',round(1/iterFac*NrepTrk0)) };
      
      PtJitFac0 = sPrm0.ROOT.CPR.Replicates.PtJitterFac;
      s.PtJitFac_up = { sprintf('.ROOT.CPR.Replicates.PtJitterFac=%d',2*PtJitFac0) };
      s.PtJitFac_dn = { sprintf('.ROOT.CPR.Replicates.PtJitterFac=%d',round(PtJitFac0/1.2)) };
      
      s.BoxJit_lo = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=12' };
      s.BoxJit_md = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=16' };
      s.BoxJit_hi = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=24' };
      
      AugUseFF0 = sPrm0.ROOT.CPR.Replicates.AugUseFF;
      s.AugUseFF_flp = { sprintf('.ROOT.CPR.Replicates.AugUseFF=%d',~AugUseFF0) };
      
      PruneSig0 = sPrm0.ROOT.CPR.Prune.DensitySigma;
      s.PruneSig_up = { sprintf('.ROOT.CPR.Prune.DensitySigma=%.3f',2*PruneSig0) };
      s.PruneSig_dn = { sprintf('.ROOT.CPR.Prune.DensitySigma=%.3f',1/2*PruneSig0) };
    end
    
    function writePchDir(pchdir,s)
      if exist(pchdir,'dir')==0
        fprintf(1,'Making pch dir: %s\n',pchdir);
        [succ,msg] = mkdir(pchdir);
        if ~succ
          error('Failed to create dir: %s\n',msg);
        end
      end
      
      fns = fieldnames(s);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        fname = fullfile(pchdir,[f '.m']);
        cellstrexport(s.(f),fname);
        fprintf(1,'Wrote %s\n',fname);
      end
    end
    
    function printPrmPchDir(pchdir)
      dd = dir(fullfile(pchdir,'*.m'));
      nn = {dd.name}';
      for n=nn(:)',n=n{1}; %#ok<FXSET>
        fname = fullfile(pchdir,n);
        [~,fnameS] = fileparts(fname);
        fprintf(1,'%s\n',fnameS);
        l = readtxtfile(fname);
        fprintf(1,'%s\n',l{:});
      end
    end
    
    function printPrmPchs(s)
      fns = fieldnames(s);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        fprintf(1,'%s\n',f);
        disp(s.(f));
        fprintf(1,'\n');
      end
    end
    
    function tblres = pchScores(xverr,pchNames,varargin)
      [ipchBase,ptilesImprove,nptsImproveThresh] = myparse(varargin,...
        'ipchBase',1, ... % reference/base patch
        'ptilesImprove',[60 90], ...
        'nptsImproveThresh',8 ...
        );
      
      [nXV,npts,npch] = size(xverr);
      
      %% normalized err; for each pt, scale by the median 'base' err
      xverrbasemedn = median(xverr(:,:,ipchBase),1);
      xverrnorm = xverr./xverrbasemedn;
      
      assert(all(xverrnorm(:)>0 | isnan(xverrnorm(:))));
      nptiles = numel(ptilesImprove);
      errptl = prctile(xverrnorm,ptilesImprove,1); % [nptiles x npts x npch]
      
      % compute fractional change in ptile values from base, per pt
      % 0=>no change in ptile
      % pos=>reduced err
      % neg=>increased err
      scorepts = (errptl(:,:,ipchBase)-errptl)./errptl(:,:,ipchBase);
      szassert(scorepts,[nptiles npts npch]);      
      
      % pt is improved if it is improved for all ptiles
%      ptimproved = scorepts>0;
%      nptimprove = squeeze(sum(sum(ptimproved,1),2)); %[npch]
      ptimproved = all(scorepts>0,1); % every ptile
      ptimproved = squeeze(ptimproved)'; 
      szassert(ptimproved,[npch npts]);
      ptimprovedfull = scorepts>0;
      
      % net score: score averaged over all pts, ptiles, per patch
      % ptiles that have larger relative fluctuations get more weight
      score = sum(sum(scorepts,1),2)/npts/nptiles*100;
      score = squeeze(score); % [npch x 1]. %age relative change in ptile for each pt
            
      nptimprove = sum(ptimproved,2);
      nptimprovedfull = squeeze(sum(sum(ptimprovedfull,1),2));
      pch = pchNames;
      tblres = table(score,nptimprove,nptimprovedfull,pch);
      [~,idx] = sort(score,'descend');
      tblres = tblres(idx,:);
    end
    
    function sPrm = readApplyPatch(sPrm,pchfile,varargin)
      verbose = myparse(varargin,...
        'verbose',true);
      
      pchs = readtxtfile(pchfile,'discardemptylines',true);
      npch = numel(pchs);
      if verbose
        fprintf(1,'Read patch file %s. %d patches.\n',pchfile,npch);
      end
      for ipch=1:npch
        pch = pchs{ipch};
        pch = ['sPrm' pch ';']; %#ok<AGROW>
        tmp = strsplit(pch,'=');
        pchlhs = strtrim(tmp{1});
        if verbose
          fprintf(1,'  patch %d: %s\n',ipch,pch);
          fprintf(1,'  orig (%s): %s\n',pchlhs,evalc(pchlhs));
        end
        eval(pch);
        if verbose
          fprintf(1,'  new (%s): %s\n',pchlhs,evalc(pchlhs));
        end
      end
    end
    
    function genNewPrmFile(basePrmFile,newPrmFile,pchDir,pchSel)
      % Create/save a new parameter file
      %
      % basePrmFile: char, base/starting parameter struct
      % newPrmFile: char, save new params to this file
      % pchDir: patch dir
      % pchSel: cellstr, patches in pchDir to apply
      
      sPrm = loadSingleVariableMatfile(basePrmFile);
      fprintf(1,'Loaded base parameters from %s.\n',basePrmFile);
      sPrm0 = sPrm;
      
      npchsel = numel(pchSel);
      for ipchsel=1:npchsel
        pchfile = fullfile(pchDir,pchSel{ipchsel});
        if ~strcmp(pchfile(end-1:end),'.m')
          pchfile = [pchfile '.m']; %#ok<AGROW>
        end
        sPrm = HPOptim.readApplyPatch(sPrm,pchfile);        
      end
      
      leap.structdiff(sPrm0,sPrm);
      
      if exist(newPrmFile,'file')>0
        error('File ''%s'' exists.',newPrmFile);
      end
      save(newPrmFile,'-mat','sPrm');
      fprintf(1,'Wrote new parameter file to %s.\n',newPrmFile);
    end
    
  end
    
    %     function hpoptimxv(lObj,xvTbl,xvSplt)
    %       assert(islogical(xvSplt) && ismatrix(xvSplt) && size(xvSplt,1)==height(xvTbl));
    %       error('Expected split definition to be a logical matrix with %d rows.\n',...
    %         height(xvTbl));
    %
    %       kfold = size(xvSplt,2);
    %       wbObj = WaitBarWithCancelCmdline;
    %       xvArgs = {'kfold' kfold 'wbObj' wbObj 'tblMFgt' xvTbl 'partTst' xvSplt};
    %
    %       % base case
    %       sPrm0 = lObj.trackGetTrainingParams();
    %       lObj.trackCrossValidate(xvArgs{:});
    %       xv0 = lObj.xvResults;
    %
    %       while 1
    %         % generate patches
    %         paramDels = HPOptim.genPrmDeltas(sPrm0);
    %         xvDels = structfun(@(x)lclRunWithPrm(lObj,x),paramDels,'uni',0);
    %         % xvDels is [nDel x 1] cell of xv results
    %
    %         % compute scores
    %
    %         % look where going down on some params wouldn't have hurt
    %
    %         % select those deltas where
    %         % - >= nptsThresh (8) pts were improved
    %         sPrm0 = lclGenNewParams();
    %         lObj.trackSetTrainingParams(sPrm0);
    %
    %         % save state
    %
    %         % stopping criterion
    %         if stop
    %           break;
    %         end
    %       end
    %
    %       % lObj has latest params
    %     end
    
    %
    %
    %
    % if tfPPatch
    %       [~,paramPatchFileS,~] = fileparts(paramPatchFile);
    %       patches = readtxtfile(paramPatchFile);
    %       npatch = numel(patches);
    %       fprintf(1,'Read parameter patch file %s. %d patches.\n',paramFile,...
    %         npatch);
    %       sPrm = lObj.trackGetTrainingParams();
    %       for ipch=1:npatch
    %         pch = patches{ipch};
    %         pch = ['sPrm' pch ';']; %#ok<AGROW>
    %         tmp = strsplit(pch,'=');
    %         pchlhs = strtrim(tmp{1});
    %         fprintf(1,'  patch %d: %s\n',ipch,pch);
    %         fprintf(1,'  orig (%s): %s\n',pchlhs,evalc(pchlhs));
    %         eval(pch);
    %         fprintf(1,'  new (%s): %s\n',pchlhs,evalc(pchlhs));
    %       end
    %       lObj.trackSetTrainingParams(sPrm);
    %       outfileBase = [outfileBase '_' paramPatchFileS];
    %     end
    %     outfileBase = [outfileBase '_' datestr(now,'yyyymmddTHHMMSS')];
    %
    %
    %     savestuff = struct();
    %     savestuff.sPrm = lObj.trackGetTrainingParams();
    %     savestuff.xvArgs = xvArgs;
    %     savestuff.xvRes = lObj.xvResults;
    %     savestuff.xvResTS = lObj.xvResultsTS; %#ok<STRNU>
    %     outfile = fullfile(lblP,[outfileBase '.mat']);
    %     fprintf('APTCluster: saving xv results: %s\n',outfile);
    %     save(outfile,'-mat','-struct','savestuff');
    % end
  
  
end