classdef MAGT
  
  methods (Static)
    
    function report(tblRes,lObj,imreadfcn,varargin)
      % Compare MA preds to lbls for xv, gt etc
      
      t = tblRes;
      % t.mov = MovieIndex(t.mov);
      NMONTAGEPLOTMAX = 240; % = (20 pages) * 3x4 montage

      [nmontage,fcnAggOverPts,aggLabel] = myparse(varargin,...
        'nmontage',min(NMONTAGEPLOTMAX,height(t)),...
        'fcnAggOverPts',@(x)max(x,[],2), ... % or eg @mean
        'aggLabel','Max' ...
        );      
            
      clrs = lObj.LabelPointColors;
      nclrs = size(clrs,1);
      
      errsCell = tblRes.matchcosts;
      errs = cat(1,errsCell{:});
      movs = arrayfun(@(zmov,znmatches)repmat(zmov,znmatches,1),...
        tblRes.mov,tblRes.numMatch,'uni',0);
      movs = cat(1,movs{:});
      npts = size(errs,2);
      errsAgg = fcnAggOverPts(errs);
      
      % Err by landmark. Equal weight per match
      h = figure('Name','Err by landmark');
      ax = axes;
      boxplot(errs,'colors',clrs,'boxstyle','filled');
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'Err by landmark',args{:});
      ax.YGrid = 'on';
      
      % AvErrAcrossPts by movie. Equal weight per match
      tstr = sprintf('%s (over landmarks) GT err by movie',aggLabel);
      h(end+1,1) = figurecascaded(h(end),'Name',tstr);
      ax = axes;
      %iMovAbs = t.mov;
      % [iMovAbs,gt] = t.mov.get;
      % assert(all(gt));
      grp = categorical(movs);
      grplbls = arrayfun(@(z1,z2)sprintf('mov%s (n=%d)',z1{1},z2),...
        categories(grp),countcats(grp),'uni',0);
      boxplot(errsAgg,grp,'colors',clrs,'boxstyle','filled',...
        'labels',grplbls);
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Movie',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,tstr,args{:});
      ax.YGrid = 'on';
      
      % Mean err by movie, pt
      h(end+1,1) = figurecascaded(h(end),'Name','Mean GT err by movie, landmark');
      ax = axes;
      tblTmp = table(movs,errs,'VariableNames',{'mov' 'errs'});
      tblStats = grpstats(tblTmp,{'mov'});
      %tblStats.mov = tblStats.mov.get;
      tblStats = sortrows(tblStats,{'mov'});
      movUnCnt = tblStats.GroupCount; % [nmovx1]
      meanL2Err = tblStats.mean_errs; % [nmovxnpt]
      nmovUn = size(movUnCnt,1);
      szassert(meanL2Err,[nmovUn npts]);
      meanL2Err(:,end+1) = nan; % pad for pcolor
      meanL2Err(end+1,:) = nan;
      hPC = pcolor(meanL2Err);
      hPC.LineStyle = 'none';
      colorbar;
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'Movie',args{:});
      xticklbl = arrayfun(@num2str,1:npts,'uni',0);
      yticklbl = arrayfun(@(x)sprintf('mov%d (n=%d)',x,movUnCnt(x)),1:nmovUn,'uni',0);
      set(ax,'YTick',0.5+(1:nmovUn),'YTickLabel',yticklbl);
      set(ax,'XTick',0.5+(1:npts),'XTickLabel',xticklbl);
      axis(ax,'ij');
      title(ax,'Mean GT err (px) by movie, landmark',args{:});   

      MAGT.trackLabelMontage(lObj,tblRes,'nplot',nmontage,...
        'readImgFcn',imreadfcn);
    end

    function errscore = montageErrScore(t)
      n = height(t);
      errscore = nan(n,1);
      
      matchscore = zeros(n,1);
      tfmatch = t.numMatch>0;
      matchscore(tfmatch) = cellfun(@(x)mean(x(:)),t.matchcosts(tfmatch));
      % matchscore(~tfpred) will be zero
      FNERR = 50000; % something big
      FPERR = 10000; % big, but not as bad as a missed detect
      errscore = FNERR*t.numFN + FPERR*t.numFP + matchscore;      
    end
    
    function [tblRes,I,tfReadFailed,tblCocoIms] = readCoco(tblRes,splitProjDirs)
      splitValTFs = fullfile(splitProjDirs,'val_TF.json');
      cos = cellfun(@Coco,splitValTFs);
      tblCocoIms = arrayfun(@(x)struct2table(x.j.images),cos,'uni',0);
      
      t = tblRes;
      n = height(t);
      I = cell(n,1);
      tfReadFailed = false(n,1);
      for i=1:n
        fold = t.fold(i);
        frm = t.frm(i);
        tCoco = tblCocoIms{fold};
        j = find(t.mov(i)-1==tCoco.movid & t.frm(i)-1==tCoco.frm & t.iTgt(i)-1==tCoco.patch);
        tfReadFailed(i) = numel(j)~=1;
        if tfReadFailed(i)
          warningNoTrace('Image for split %d, movie %d, frame %d, tgt %d not found in db.',...
            fold,t.mov(i),t.frm(i),t.iTgt(i));
          continue;
        end
        
        imfname = tblCocoIms{fold}.file_name{j};
        im = imread(imfname);
        I{i} = im;
      end
    end
    
    function [d,match,matchcosts,unmatchedlbls,unmatchedprds,nFP,nFN,nMch,nLbl,nPrd] = ...
        comparePredsLbls(ploc,pprd,lblexist,prdexist)
      % Compare MA preds to lbls for xv, gt etc
      %
      % ploc: [N x maxNTgt x npt x 2]. keypt locs
      % pprd: "
      % lblexist: [N x maxNTgt]. logical flags for whether keypt exists (was
      %   labeled or predicted at all)
      % prdexist: "
      %
      % d: [N] cell. d{i} is a nLblExist(i) x nPrdExist(i) array of distances
      %   between shapes; rows correspond to lbl-shapes and cols to
      %   prd-shapes. Currently the distance is euclidean
      % match: [N] cell. match{i} is a [nMatch(i) x 2] array indicating matches
      %   between rows (indexing lbl shapes) and cols (indexing prd shapes)
      % matchcosts: [N] cell. matchcosts{i} is [nMatch(i) x npt] distances from
      %   d{i} for matches/pts in match{i}
      % unmatchedlbls: [N] cell. unmatchedlbls{i} is a vector of indices for
      %   rows (lbl shapes) with no matched/corresponding pred
      % unmatchedprds: [N] cell. unmatchedprds{i} is a vector of indices for
      %   cols (prd shapes) with no matched/corresponding lbl
      % nFP: [N] double. number of false positives (preds without a match)
      % nFN: [N] etc
      % nMch: [N] double. number of matches
      % nPrd: [N] double. number of labeled shapes
      % nLbl: [N] double. etc
      
      UNMATCHEDCOST = 999;
      
      [N,maxNtgt,npt,dim] = size(ploc);
      nptdim = npt*dim;
      d = cell(N,1);
      match = cell(N,1);
      matchcosts = cell(N,1);
      %costs = cell(N,1);
      unmatchedlbls = cell(N,1);
      unmatchedprds = cell(N,1);
      nFP = nan(N,1);
      nFN = nan(N,1);
      nMch = nan(N,1);
      nLbl = nan(N,1);
      nPrd = nan(N,1);
      for i=1:N
        lblE = lblexist(i,:);
        n0 = nnz(lblE);
        p0 = ploc(i,lblE,:,:);
        p0 = reshape(p0,n0,npt,dim);
        p0 = permute(p0,[1,3,2]); % n x dim x npt
        
        lblP = prdexist(i,:);
        n1 = nnz(lblP);
        p1 = pprd(i,lblP,:,:);
        p1 = reshape(p1,n1,npt,dim);
        p1 = permute(p1,[1,3,2]);
        
        pdistpts = nan(n0,n1,npt);
        for ipt=1:npt
          pdistpts(:,:,ipt) = pdist2(p0(:,:,ipt),p1(:,:,ipt),'euclidean');
        end
        d{i} = sum(pdistpts,3)/npt; % L2 err, avgd over pts % pdist2(p0,p1,'euclidean');
        %szassert(d{i},[n0 n1]);
        [matchtmp,uar0,uac0] = matchpairs(d{i},UNMATCHEDCOST);
        
        matchidx0 = matchtmp(:,1);
        matchidx1 = matchtmp(:,2);
        %costs{i} = d{i}(sub2ind(size(d{i}),matchidx0,matchidx1));
        %costs{i} = costs{i}(:)';
        % more compact style of matching vector eg produced by munkres
        match{i} = zeros(1,n0);
        match{i}(matchidx0) = matchidx1;
        nmatch = numel(matchidx0);
        matchcosts{i} = nan(nmatch,npt);
        for imatch=1:nmatch
          % sum over L2 distances between all pts
          matchcosts{i}(imatch,:) = pdistpts(matchidx0(imatch),matchidx1(imatch),:);
        end
        
        %   [m1,c1] = munkres(d2)
        
        %   nMatch = size(m0,1);
        unmatchedlbls{i} = uar0;
        unmatchedprds{i} = uac0;
        nFP(i) = numel(uac0); % pred with no lbl
        nFN(i) = numel(uar0); % lbl with no pred
        nMch(i) = nmatch;
        nLbl(i) = n0;
        nPrd(i) = n1;
      end
      
    end
    
    function trackLabelMontage(obj,tbl,varargin)
      [nr,nc,h,npts,nphyspts,nplot,frmlblclr,frmlblbgclr,readImgFcn] = ...
        myparse(varargin,...
        'nr',3,...
        'nc',4,...
        'hPlot',[],...
        'npts',obj.nLabelPoints,... % hack
        'nphyspts',obj.nPhysPoints,... % hack
        'nplot',height(tbl),... % show/include nplot worst rows
        'frmlblclr',[1 1 1], ...
        'frmlblbgclr',[0 0 0], ...
        'readImgFcn',@obj.trackLabelMontageProcessData ...
        );
      
      if nplot>height(tbl)
        warningNoTrace('''nplot'' argument too large. Only %d GT rows are available.',height(tbl));
        nplot = height(tbl);
      end
      
      tbl.escore = MAGT.montageErrScore(tbl);
      tbl = sortrows(tbl,{'escore'},{'descend'});
      tbl = tbl(1:nplot,:);
      
      [tbl,I,tfReadFailed] = readImgFcn(tbl);
      
      I = cellfun(@DataAugMontage.convertIm2Double,I,'uni',0);
      
      %tblPostRead = tbl(:,{'pLbl' 'pTrk' 'mov' 'frm' 'iTgt' errfld});
      tblPostRead = tbl;
      tblPostRead(tfReadFailed,:) = [];
      
      nread = height(tblPostRead);
      frmLblsAll = cell(nread,1);
      for i=1:nread
        s = tblPostRead(i,:);
        if s.numFN>0
          frmLblsAll{i} = sprintf('mov/frm/tgt=%d/%d/%d,numFN=%d',s.mov,s.frm,s.iTgt,s.numFN);
        elseif s.numFP>0
          frmLblsAll{i} = sprintf('mov/frm/tgt=%d/%d/%d,numFP=%d',s.mov,s.frm,s.iTgt,s.numFP);
        else
          frmLblsAll{i} = sprintf('mov/frm/tgt=%d/%d/%d,err=%.2f',s.mov,s.frm,s.iTgt,s.escore);
        end
      end
      
      [~,maxntgt,npt,d] = size(tblPostRead.p);
      tblPostRead.p = reshape(permute(tblPostRead.p,[1 3 4 2]),nread,npt*d,maxntgt);
      tblPostRead.pLbl = reshape(permute(tblPostRead.pLbl,[1 3 4 2]),nread,npt*d,maxntgt);
      
      pppi = obj.predPointsPlotInfo;
      plotClrs = pppi.Colors;
      nrowsPlot = height(tblPostRead);
      startIdxs = 1:nr*nc:nrowsPlot;
      for i=1:numel(startIdxs)
        plotIdxs = startIdxs(i):min(startIdxs(i)+nr*nc-1,nrowsPlot);
        frmLblsThis = frmLblsAll(plotIdxs);
        for iView=1:obj.nview
          h(end+1,1) = figure('Name','Tracking Error Montage','windowstyle','docked'); %#ok<AGROW>
          pColIdx = (1:nphyspts)+(iView-1)*nphyspts;
          pColIdx = [pColIdx pColIdx+npts]; %#ok<AGROW>
          Shape.montage(I(:,iView),tblPostRead.pLbl(:,pColIdx,:),'fig',h(end),...
            'nr',nr,'nc',nc,'idxs',plotIdxs,...
            'colors',plotClrs,...
            'framelbls',frmLblsThis,'framelblscolor',frmlblclr,...
            'framelblsbgcolor',frmlblbgclr,'p2',tblPostRead.p(:,pColIdx,:),...
            'p2marker','+','titlestr','Tracking Montage, descending err (''+'' is tracked)');
        end
      end      
    end
    
  end
  
end