classdef GTPlot
  methods (Static)
    
    function [n,npts,nvw,nsets,l2errptls,l2mnerrptls,ntype] = ...
        errhandler(err,ptiles,varargin)
      % err: Either:
      %  1. numeric array [n x npts x (x/y) x nviews x nsets]
      %  2. cell array [nsets] where err{iset} is [n_iset x npts x (x/y) x nviews]
      %  3. cell array [nviews] where err{ivw} is [n_ivw x npts x (x/y) x nsets]
      %     (set cellWithViews==true)
      % ptiles: [nptl] vector of ptiles
      % 
      % The 3rd dim (x/y) could be just (x) or (x/y/z) etc.
      %
      % "views" and "sets" are really just arbitrary dimensions for tiling 
      % plots. In ptiles plots, sets are shown along the x-axis of each 
      % plot, showing eg a progression of improvement in err. These might
      % be better called "rounds". Meanwhile, views are tiled (via 
      % subplots) in the y-direction.
      %
      % n: If 1, the common number of rows.
      %    If 2, an [nsets] array.
      %    If 3, an [nviews] array.
      % l2errptls: [nptl x npts x nvw x nsets] l2 err ptile matrix
      % l2mnerrptls: [nptl x nvw x nsets] ptiles of mean-err-over-pts 
      % ntype: either 'scalar', 'perset', 'perview'
      
      cellPerView = myparse(varargin,...
        'cellPerView',false...
        );
            
      if iscell(err)
        err = err(:);
        if cellPerView
          nvw = numel(err);
          [n,npts,d,nsets] = cellfun(@size,err);
          assert(all(npts==npts(1)));
          assert(all(d==d(1)));
          assert(all(nsets==nsets(1)));          
          npts = npts(1);
          %d = d(1);
          nsets = nsets(1);
          ntype = 'perview';
        else
          nsets = numel(err);
          [n,npts,d,nvw] = cellfun(@size,err);
          assert(all(npts==npts(1)));
          assert(all(d==d(1)));
          assert(all(nvw==nvw(1)));
          npts = npts(1);
          % d=d(1)
          nvw = nvw(1);
          ntype = 'perset';
        end
      elseif isnumeric(err)
        [n,npts,d,nvw,nsets] = size(err);
        ntype = 'scalar';
      else
        assert(false);
      end
      
      nptl = numel(ptiles);      
      l2errptls = nan(nptl,npts,nvw,nsets);
      l2mnerrptls = nan(nptl,nvw,nsets);
      for ivw=1:nvw
      for iset=1:nsets
        if iscell(err)
          if cellPerView
            errSetVw = err{ivw}(:,:,:,iset);
          else
            errSetVw = err{iset}(:,:,:,ivw);
          end
        else
          errSetVw = err(:,:,:,ivw,iset);
        end
        % errSetVw is [n_set_view x npts x d]
        l2errSV = sqrt(sum(errSetVw.^2,3)); % [n_set_view x npts]
        l2errptls(:,:,ivw,iset) = prctile(l2errSV,ptiles,1);
        l2errSetMean = sum(l2errSV,2)/npts;
        assert(iscolumn(l2errSetMean));
        l2mnerrptls(:,ivw,iset) = prctile(l2errSetMean,ptiles,1);
      end
      end
    end
    
    function [hFig,hAxs] = bullseyePtiles(err,I,xyLbl,varargin)
      % Bullseyes overlaid on images
      %
      % I: [nviews] cell array of images to use
      % xyLbl: [npt x 2 x nview] GT labels where bullseyes will be centered
      % err: [n x npts x nviews x nsets] Error array. 
      %  - n is the number of rows/frames
      %  - npts is the number of landmarks
      %  - nviews can be 1 for single-view data. "views" and "sets" are 
      % really just arbitrary dimensions for tiling bullseye plots 
      %  - nsets can be 1 for eg a single XV run, or greater than 1 when 
      %  eg a parameter is titrated it is desired to compare multiple runs.
      %
      % hFig: figure handle
      % hAxs: [nviews x nsets] axes handles
      
      [ptiles,hFig,xyLblPlotArgs,setNames,ptileCmap,lineWidth,...
        contourtype,titlefontsize,...
        doleg,legfontsize,legstrs,...
        trusz] = ...
        myparse(varargin,...
        'ptiles',[50 75 90 95 97.5 99 99.5],...
        'hFig',[],...
        'xyLblPlotArgs',{'m.' 'markersize' 20},...
        'setNames',[],...
        'ptileCmap','lines',...
        'lineWidth',1,...
        'contourtype','circle',... % either 'circle','ellipse','arb'. 
          ... % If ellipse or arb, then err should have size 
          ... % [n x npts x 2 x nviews x nsets] (3rd dim = x/y)
        'titlefontsize',22, ...
        'doleg',true,...
        'legfontsize',16, ...
        'legstrs',[],...
        'trusz',true ...
        );
      
      [n,npts,nviews,nsets,l2errptls,l2mnerrptls] = ...
        GTPlot.errhandler(err,ptiles);
      
      assert(isvector(I) && iscell(I) && numel(I)==nviews);
      szassert(xyLbl,[npts,2,nviews]);
      if isempty(setNames)
        setNames = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setNames) && numel(setNames)==nsets);
           
      if isempty(hFig)
        hFig = figure();
        set(hFig,'Color',[1 1 1]);
      else
        figure(hFig);
        clf
      end
      
      nptiles = numel(ptiles);
      %err_prctiles = nan(nptiles,npts,nviews,nsets);
      gaussfitIfos = cell(npts,nviews,nsets);
      switch contourtype
        case 'ellipse'
          for ipt=1:npts
            for ivw=1:nviews
              for iset=1:nsets
                if iscell(err)
                  xyerr = squeeze(err{iset}(:,ipt,:,ivw));
                else
                  xyerr = squeeze(err(:,ipt,:,ivw,iset)); 
                end
                assert(size(xyerr,2)==2); % 2d only, xyerr should be [n_iset x 2]
                gaussfitIfos{ipt,ivw,iset} = GTPlot.gaussianFit(xyerr);
              end
            end
          end
      end
      
      colors = feval(ptileCmap,nptiles);
      hAxs = createsubplots(nviews,nsets,[.01 .01;.05 .01]);
      hAxs = reshape(hAxs,[nviews,nsets]);
      
      h = nan(1,nptiles);
      for viewi = 1:nviews
        im = I{viewi};
        xyL = xyLbl(:,:,viewi);
        
        for k = 1:nsets
          ax = hAxs(viewi,k);
          imagesc(im,'Parent',ax);
          colormap gray
          axis(ax,'image','off');
          hold(ax,'on');

          if viewi==1
            tstr = setNames{k};
            if k==1 && isscalar(n)
              tstr = sprintf('N=%d. %s',n,tstr);
            end
            title(ax,tstr,'fontweight','bold','fontsize',titlefontsize,...
              'interpreter','none');
          end
          
          if viewi==nviews && ~isscalar(n)
            xlblstr = sprintf('n=%d',n(k));
            xlabel(ax,xlblstr,'fontweight','bold','fontsize',titlefontsize,...
              'interpreter','none');
            set(ax,'Visible','on','XTickLabel',[],'YTickLabel',[]);
          end
          
          for p = 1:nptiles
            for l = 1:npts
              switch contourtype
                case 'circle'
                  rad = l2errptls(p,l,viewi,k);                  
                  h(p) = drawellipse(xyL(l,1),xyL(l,2),0,rad,rad,...
                    'Color',colors(p,:),'Parent',ax,'lineWidth',lineWidth);
                case 'ellipse'
                  gIfo = gaussfitIfos{l,viewi,k};
                  scatterpts = (l==2) & p==1;
                  h(p) = GTPlot.gaussianFitDrawContours(gIfo,...
                    xyL(l,1),xyL(l,2),ptiles(p),'scatterpts',scatterpts,...
                    'Color',colors(p,:),'Parent',ax,'lineWidth',lineWidth);
                  if p==1
                    fprintf(1,'Vw%d set%d pt%d. gIfo.mean: %s\n',...
                      viewi,k,l,mat2str(gIfo.mean));
                  end
                case 'arb'
                  assert(false,'Unimplemented.');
              end
            end
          end
          
          % do this last so it is most on top
          plot(ax,xyL(:,1),xyL(:,2),xyLblPlotArgs{:});

        end
      end
            
      if doleg
        if ~isempty(legstrs)
          assert(numel(legstrs)==nptiles);          
        else
          legstrs = cell(1,nptiles);
          for p = 1:nptiles
            legstrs{p} = sprintf('%sth %%ile',num2str(ptiles(p)));
          end
        end
        hl = legend(h,legstrs);
        set(hl,'Color','k','TextColor','w','EdgeColor','w',...
          'fontsize',legfontsize);
      end
      if trusz
        truesize(hFig);
      end
    end
    
    function [hFig,hAxs] = errCDF(err,varargin)
      % "Fraclessthan" cumulative error curves
      %
      % err: see bullseyePtiles      
      %
      % hFig: figure handle
      % hAxs: [nviews x nsets] axes handles. These are all linkaxes-ed in
      % the x-direction, so they will share common x axis lims.
      
      [n,npts,nviews,nsets] = size(err);
      
      [hFig,lineArgs,setNames,setCmap] = ...
        myparse(varargin,...
        'hFig',[],...
        'lineArgs',{'-','linewidth',1.5},...
        'setNames',[],...
        'setCmap','jet');
      
      if isempty(hFig)
        hFig = figure();
        set(hFig,'Color',[1 1 1]);
      else
        figure(hFig);
        clf
      end
      
      if isempty(setNames)
        setNames = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setNames) && numel(setNames)==nsets);      
      
      fracleqerr = cell(npts,nviews,nsets);
      for l = 1:npts
        for v = 1:nviews
          for p = 1:nsets
            sortederr = sort(err(:,l,v,p));
            [sortederr,nleqerr] = unique(sortederr);
            fracleqerr{l,v,p} = cat(2,nleqerr./size(err,1),sortederr);
            %minerr = min(minerr,fracleqerr{l,v,p}(find(fracleqerr{l,v,p}(:,1)>=minfracplot,1),2));
          end
        end
      end
            
      hAxs = createsubplots(nviews,npts,[.05 0;.1 .1]);
      hAxs = reshape(hAxs,[nviews,npts]);
      
      clrs = feval(setCmap,nsets);
      for l = 1:npts
        for v = 1:nviews
          ax = hAxs(v,l);
          hold(ax,'on');
          grid(ax,'on');
          
          tfPlot1 = v==1 && l==1;
          h = gobjects(nsets,1);
          for p=1:nsets
            h(p) = plot(ax,fracleqerr{l,v,p}(:,2),fracleqerr{l,v,p}(:,1),...
              lineArgs{:},'color',clrs(p,:));
          end
          tstr = sprintf('vw%d pt%d',v,l);
          if l==1 && v==1
            tstr = sprintf('N=%d. %s',n,tstr);
          end
          title(ax,tstr,'fontweight','bold','fontsize',16);

          
          %     if l == 1 && v == 1,
          %       legend(h,prednames,'Location','southeast');
          %     end
          %     title(hax(v,l),sprintf('%s, %s',lbld.cfg.LabelPointNames{l},lbld.cfg.ViewNames{v}));
          
          set(ax,'XScale','log');
          if tfPlot1
            title(ax,tstr,'fontweight','bold');
            xlabel(ax,'Error (raw,  px)','fontsize',14);
            ylabel(ax,'Frac. smaller','fontsize',14);
            
            if nsets>1
              legend(h,setNames,'location','southeast');
            end
          else
            set(ax,'XTickLabel',[],'YTickLabel',[]);
          end
        end
      end
      
      linkaxes(hAxs(:),'x');
      %xlim(hAxs(1),[1 32]);
    end
    
    function [hFig,hAxs] = ptileCurves(err,varargin)
      % Constant-percentile curves over a titration set
      %
      % err: see errHandler
      %
      % hFig: figure handle
      % hAxs: [nviews x nsets] axes handles
      
      [ptiles,hFig,lineArgs,setNames,axisArgs,ptnames,...
        createsubplotsborders,titleArgs,errCellPerView,viewNames,...
        legendFontSize,errbars,errbarptiles] = ...
        myparse(varargin,...
        'ptiles',[50 75 90 95 97.5 99 99.5],...
        'hFig',[],...
        'lineArgs',{'markersize',20},...
        'setNames',[],...
        'axisArgs',{'XTicklabelRotation',45,'FontSize' 20},...
        'ptnames',[],...
        'createsubplotsborders',[.05 0;.12 .12],...
        'titleArgs',{'fontweight','bold'},...
        'errCellPerView',false,...
        'viewNames',[],... % [nview] cellstr
        'legendFontSize',10, ...
        'errbars',[], ... % [nptl x (npts+1) x nvw] err bars spec'd per 
                          ... % (pt,vw,ptl). errbar shown is +/- this qty.
                          ... % errbars are constant across sets. 
                          ... % The last column of errbars is for the 
                          ... % "mean over all pts"
        'errbarptiles',[] ... % [nptl] ptiles used for errbars. must match 
                          ... % ptiles.
        );
      
      [ns,npts,nviews,nsets,l2errptls,l2mnerrptls,nstype] = ...
        GTPlot.errhandler(err,ptiles,'cellPerView',errCellPerView);
      
      if isempty(ptnames)
        ptnames = arrayfun(@(x)sprintf('pt%d',x),1:npts,'uni',0);
      end
      assert(iscellstr(ptnames) && numel(ptnames)==npts);
      
      if isempty(hFig)
        hFig = figure();
        set(hFig,'Color',[1 1 1]);
      else
        figure(hFig);
        clf;
      end
      
      if isempty(setNames)
        setNames = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setNames) && numel(setNames)==nsets);
      setNames = setNames(:);
      if iscell(err) && strcmp(nstype,'perset')
        setNames = arrayfun(@(x,y)sprintf('%s (n=%d)',x{1},y),setNames,ns,'uni',0);
      end
      
      if isempty(viewNames)
        viewNames = arrayfun(@(x)sprintf('View %d',x),1:nviews,'uni',0);
      end
      assert(iscellstr(viewNames) && numel(viewNames)==nviews);
      if iscell(err) && strcmp(nstype,'perview')
        viewNames = arrayfun(@(x,y)sprintf('%s (n=%d)',x{1},y),viewNames(:),ns(:),'uni',0);
      end
      
      tfEB = ~isempty(errbars);
      if tfEB
        assert(isequal(errbarptiles,ptiles));
        assert(ndims(errbars)<=3);
        if isequal(size(errbars,1,2,3),[numel(ptiles) npts+1 nviews])
          % none
        elseif isequal(size(errbars,1,2,3),[numel(ptiles) npts nviews])
          % add nans for "all means" errbars
          warningNoTrace('Adding nans for "mean over all pts" errbars');
          errbars(:,end+1,:) = nan;          
        else
          assert(false,'Errbar info has wrong size.');
        end
        
        %l2errptls: [nptl x npts x nvw x nsets]
        l2errptlsmu = mean(l2errptls,4);
        l2errptlsdelmu = l2errptls-l2errptlsmu;
        l2errptlsZS = l2errptlsdelmu./errbars(:,1:npts,:); 
        % [nptl npts nvw nsets]
      end

      hAxs = createsubplots(nviews,npts+1,createsubplotsborders);
      hAxs = reshape(hAxs,nviews,npts+1);
      for ivw=1:nviews
        for ipt=[1:npts inf]
          tfPlot1Top = ivw==1 && ipt==1;
          tfPlot1Bot = ivw==nviews && ipt==1;
          
          % Get/Compute: 
          % y: [nptls x nsets] err percentiles for each set
          % ax: axis in which to plot
          % tstr: title str

          if ~isinf(ipt)
            y = squeeze(l2errptls(:,ipt,ivw,:)); % [nptl x nsets]
            ax = hAxs(ivw,ipt);
            tstr = ptnames{ipt};
            if isscalar(ns) && strcmp(nstype,'scalar') && tfPlot1Top
              tstr = sprintf('N=%d\n%s',ns,tstr);
            end
          else
            y = squeeze(l2mnerrptls(:,ivw,:)); % [nptl x nsets]
            ax = hAxs(ivw,npts+1);
            tstr = sprintf('mean\nallpts'); %'mean allpts shown';
          end
          
          axes(ax);
          
          args = {...
            'YGrid' 'on' 'XGrid' 'on' 'XLim' [0 nsets+1] 'XTick' 1:nsets ...
            'XTickLabel' setNames 'TickLabelInterpreter' 'none'};
          args = [args axisArgs]; %#ok<AGROW>
            
          x = 1:nsets; % croptypes
          h = plot(x,y','.-',lineArgs{:});  % nptl curves plotted
          set(ax,args{:});
          hold(ax,'on');
          ax.ColorOrderIndex = 1;
          
          if tfEB 
            if ~isinf(ipt)
              ybar = errbars(:,ipt,ivw)'; % [1 x nptl]
            else
              ybar = errbars(:,end,ivw)'; % etc
            end
            y0 = y'-ybar; % [nsets x nptl]
            y1 = y'+ybar; % [nsets x nptl]
            for iset=1:nsets
            for iptl=1:size(y0,2)
              plot(x([iset iset]),[y0(iset,iptl) y1(iset,iptl)],':',...
                'Color',h(iptl).Color,'linewidth',3); 
            end
            end
            ax.ColorOrderIndex = 1;
          end

          if tfPlot1Top
            legstrs = strcat(numarr2trimcellstr(ptiles'),'%');
            hLeg = legend(h,legstrs);
            hLeg.FontSize = legendFontSize;
            %xlabel('Crop type','fontweight','normal','fontsize',14);
          end
          if tfPlot1Bot
            % none
          else
            set(ax,'XTickLabel',[]);
          end
          title(tstr,titleArgs{:});
          if ipt==1
            if tfPlot1Top
              ystr = sprintf('%s (raw err, px)',viewNames{1});              
            else
              ystr = viewNames{ivw};
            end
            ylabel(ystr,'fontweight','bold','interpreter','none');
          else
            set(ax,'YTickLabel',[]);
          end
        end
        
        linkaxes(hAxs(ivw,:));
        
        hAxs(ivw,1).YLim(1) = 0;
      end
      
      DOSZ = false;
      if DOSZ
        hFigZS = figure;
        hAxsZS = createsubplots(nviews,npts+1,createsubplotsborders);
        hAxsZS = reshape(hAxsZS,nviews,npts+1);
        
        for ivw=1:nviews
          for ipt=[1:npts inf]
            if ~isinf(ipt)
              % [nptl npts nvw nsets]
              y = squeeze(l2errptlsZS(:,ipt,ivw,:)); % [nptl x nsets]
              ax = hAxsZS(ivw,ipt);
              tstr = ptnames{ipt};
            else
              y = squeeze(sum(l2errptlsZS(:,:,ivw,:),2)); % [nptl x nsets]
              ax = hAxsZS(ivw,npts+1);
              tstr = sprintf('mean\nallpts'); %'mean allpts shown';
            end
            
            szassert(y,[numel(ptiles) nsets]);
            
            axes(ax);
            
            args = {...
              'YGrid' 'on' 'XGrid' 'on' 'XLim' [0 nsets+1] 'XTick' 1:nsets ...
              'XTickLabel' setNames 'TickLabelInterpreter' 'none'};
            args = [args axisArgs]; %#ok<AGROW>
            
            x = 1:nsets; % croptypes
            h = plot(x,y','.-',lineArgs{:});  % nptl curves plotted
            set(ax,args{:});
            hold(ax,'on');
            ax.ColorOrderIndex = 1;
          end
        end
      end
    end
    
    function [hFig,scores] = ptileCurvesPickBest(err,ptilesScore,varargin)
      % Like ptileCurves, but the "best" (least err) Set is picked for each
      % View. 
      %
      % ALL PTS AND PTILES ARE WEIGHTED "equally" in the score; see below
      %
      % scores: [nviews x nsets], tracking err (px), averaged over
      % pts/ptiles
      
      [viewNames,setNames,errCellPerView,fignums,figpos,ptileCurvesArgs] = ...
        myparse(varargin,...
        'viewNames',[],... 
        'setNames',[],...
        'errCellPerView',false,...
        'fignums',[11 12],...
        'figpos',[1 1 1920 960],...
        'ptileCurvesArgs',{}...
        );      
              
      
      [ns,npts,nviews,nsets,l2errptls,l2mnerrptls,nstype] = ...
        GTPlot.errhandler(err,ptilesScore,'cellPerView',errCellPerView);
      
      % Return setNames/viewNames from ptileCurves
      if isempty(setNames)
        setNames = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setNames) && numel(setNames)==nsets);
      setNames = setNames(:);
      if iscell(err) && strcmp(nstype,'perset')
        setNames = arrayfun(@(x,y)sprintf('%s (n=%d)',x{1},y),setNames,ns,'uni',0);
      end
      
      if isempty(viewNames)
        viewNames = arrayfun(@(x)sprintf('View %d',x),1:nviews,'uni',0);
      end
      assert(iscellstr(viewNames) && numel(viewNames)==nviews);
      if iscell(err) && strcmp(nstype,'perview')
        viewNames = arrayfun(@(x,y)sprintf('%s (n=%d)',x{1},y),viewNames(:),ns(:),'uni',0);
      end
      
      hFig = figure(fignums(1));
      hfig = hFig(end); 
      set(hfig,'position',figpos);
      [~,ax] = GTPlot.ptileCurves(err,...
        'hFig',hfig,...
        'viewNames',viewNames,...
        'setNames',setNames,...
        'errCellPerView',errCellPerView,...
        'ptiles',ptilesScore,...
        ptileCurvesArgs{:});
      
      szassert(l2mnerrptls,[numel(ptilesScore) nviews nsets]);      
      
      % Note on the score.
      % For each (set,view,pt), we have a distribution of errors, for which
      % we consider/plot percentiles per ptiles. We would like to compute
      % a single scalar score per (set,view) that can be minimized to yield
      % a "best" set for each view.
      %
      % To collapse info across points, we sum (average) all ptiles across
      % all points. All points are thus weighted equally. The overall
      % central tendency of the err (per se) for the points are irrelevant, 
      % as we will be minimizing the score; however, the scale of 
      % dispersions/fluctuations is important. Eg if a certain landmark has 
      % a very widely fluctuating error (ptiles that fluctate wildly over
      % sets say) then minimizing the sum-of-ptiles over sets will be
      % dominated by minimizing the error for this single point.
      %
      % The overall fluctuations of error probably scale with the central
      % tendency of the err to a first guess, so strictly speaking summing 
      % over all pts weights points-with-larger-error higher than pts with 
      % lower error. This is arguably a good thing, since we probably care 
      % more about improving the err for the worst points than points that 
      % are easy.
      %
      % To collapse across ptiles, we again sum and take a simple average.
      % For the typical case ptiles==[50 90], this means we tradeoff
      % equally eg a degradation of 1px in the median vs an improvement of
      % 1px in the 90th ptile. This is obviously very subjective and by
      % some accounts "incorrect" but for our applications this is not 
      % obviously wrong. We keep it simple and stick to the average for
      % now.      
      
      scores = squeeze(mean(l2mnerrptls,1)); 
      % scoreSR: l2 px tracking err, av over pts plotted/shown, av over 
      % ptiles plotted/shown, per view, per set
      
      hFig(end+1) = figure(fignums(2));
      hfig = hFig(end);
      set(hfig,'position',figpos);
      %set(hfig,'Name','mean err px','Position',figpos);
      axs = mycreatesubplots(nviews,1,[.05 0;.12 .12]);
      x = 1:nsets;
      for ivw=1:nviews
        ax = axs(ivw);
        hold(ax,'on');
        plot(ax,x,scores(ivw,:),'.-','markersize',20);
        if ivw==nviews
          set(ax,'XTick',x,'XTickLabel',setNames,'XTickLabelRotation',90);
        else
          set(ax,'XTick',[]);
        end
        ylabel(ax,viewNames{ivw},'fontweight','bold');
        
        idx = argmin(scores(ivw,:));
        fprintf('%s: best set is %s\n',viewNames{ivw},setNames{idx});
        plot(ax,idx,scores(ivw,idx),'ro','markersize',10,'markerfacecolor',[1 0 0]);
        
        grid(ax,'on');
        yl = ylim(ax);
        yl(1) = 0;
        ylim(ax,yl);        
      end
    end
    
    function [hFig,hAxs] = ptileCurvesZoomed(err,varargin)
      % Constant-percentile curves over a titration set; each ptile curve
      % gets its own axis
      %
      % err: [n x npts x nsets] Error array. 
      %  - n is the number of rows/frames
      %  - npts is the number of landmarks
      %  - nsets eg a parameter is titrated it is desired to compare multiple runs.
      %      
      % hFig: figure handle
      % hAxs: [nptiles x npts] axes handles
      
      [ptiles,hFig,lineArgs,ptNames,setNames,axisArgs,ylimcapbase] = ...
        myparse(varargin,...
        'ptiles',[50 75 90 95 97.5 99 99.5],...
        'hFig',[],...
        'lineArgs',{'m+'},...
        'ptNames',[],...
        'setNames',[],...
        'axisArgs',{'XTicklabelRotation',45,'FontSize' 16},...
        'ylimcapbase',false... 
        );
      
      [n,npts,nsets] = size(err);
     
      if isempty(hFig)
        hFig = figure();
        set(hFig,'Color',[1 1 1]);
      else
        figure(hFig);
        clf;
      end
      
      if isempty(setNames)
        setNames = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setNames) && numel(setNames)==nsets);

      if isempty(ptNames)
        ptNames = arrayfun(@(x)sprintf('pt%d',x),1:npts,'uni',0);
      end
      assert(numel(ptNames)==npts);

      nptiles = numel(ptiles);
      hAxs = subplots(nptiles,npts,[.06 0;.12 0]);
      for ipt=1:npts
%         if ~isinf(ipt)
          % normal branch
          errs = squeeze(err(:,ipt,:)); % nxnsets
          y = prctile(errs,ptiles); % [nptlsxnsets]
%         else
%           errs = squeeze(sum(err(:,:,ivw,:),2)/npts); % [nxnsets]
%           y = prctile(errs,ptiles); % [nptlsxnsets]
%           ax = hAxs(ivw,npts+1);
%           tstr = sprintf('vw%d, mean allpts',ivw);
%         end
        tfPlot1 = ipt==1;
        %           if tfPlot1
        %             tstr = ['XV err vs CropType: ' tstr];
        %           end
        
        args = {...
          'YGrid' 'on' 'XGrid' 'on' 'XLim' [0 nsets+1] 'XTick' 1:nsets ...
          'XTickLabel',setNames,'TickLabelInterpreter','none'};
        args = [args axisArgs];        
        x = 1:nsets;
        for iptile=1:nptiles
          ax = hAxs(iptile,ipt);
          axes(ax);
          h = plot(x,y(iptile,:)','.-','markersize',20);
          hold(ax,'on');
%           ax.ColorOrderIndex = 1;

          if iptile==1
            titleargs = {'fontweight','bold','fontsize',16};
            if ipt==1
              title(sprintf('N=%d. %s',n,ptNames{ipt}),titleargs{:});
            else
              title(ptNames{ipt},titleargs{:});
            end            
          end
          
          if ipt==1
            ptilestr = strcat(num2str(ptiles(iptile)),'%');
            ylabel(ax,ptilestr,'fontweight','bold','fontsize',16);
          else
            set(ax,'YTickLabel',[]);
          end
          set(ax,args{:});
          if ylimcapbase
            yl = ylim(ax);
            yl(2) = min(yl(2),2*y(iptile,1)); 
            ylim(ax,yl);
          end
          set(ax,'YTick',ax.YTick(1:end-1),'YTickLabel',ax.YTickLabel(1:end-1));
        end
      end
      
      for iptile=1:nptiles
        linkaxes(hAxs(iptile,:));
      end
    end

    % anisotropic bullseye utils

    function h = gaussianFitDrawContours(ifo,xc,yc,ptiles,varargin)
      % ifo: Output of .gaussianFit(xyTrkErr)
      
      [ax,meshrad,meshptsperpx,color,lineWidth,scatterpts,scatterptsjit,useextrameth] = ...
        myparse(varargin,...
        'Parent',gca,...
        'meshrad',50,...
        'meshptsperpx',1,...
        'Color',[0 0 1],...
        'lineWidth',1, ...
        'scatterpts',false,...
        'scatterptsjit',0.25,...
        'useextrameth',false ... % if true, compute/plot 2 ways (originally for debugging)
        );       

      if scatterpts
        %ifo.xy is xyTrkErr
        xyjit = ifo.xy;
        xyjit = xyjit + scatterptsjit*2*(rand(size(xyjit))-0.5);
        plot(ax,xyjit(:,1)+xc,xyjit(:,2)+yc,'.w');
      end

      if useextrameth
        % method 1: compute ptiles of actual pdf and call contour()
        xabs = linspace(xc-meshrad,xc+meshrad,2*meshrad*meshptsperpx);
        yabs = linspace(yc-meshrad,yc+meshrad,2*meshrad*meshptsperpx);
        [xabs,yabs] = meshgrid(xabs,yabs);
        ff = ifo.f([xabs(:)-xc yabs(:)-yc]);
        ff = reshape(ff,size(xabs));
        ff = -ff; % will be taking "positive" ptiles


        % These -PDF/ifo.f() values are at the various ptile contours of
        % -PDF for the dataset
        zptiles = prctile(-ifo.fxy,ptiles);
        if isscalar(zptiles)
          zptiles = [zptiles zptiles];
        end
        [~,h] = contour(ax,xabs,yabs,ff,zptiles,...
          'color',color,'linewidth',lineWidth+1,'LineStyle',':');
      end
            
      % method 2: compute ptiles of maha dist
      mhd = sqrt(ifo.mhd2xy);
      mhdPtiles = prctile(mhd,ptiles);
      nptiles = numel(ptiles);
      [atmp,btmp,theta] = cov2ell(ifo.cov); % a, b are 2sigma
      siga = atmp/2;
      sigb = btmp/2;
      
      axes(ax);
      
      h = gobjects(nptiles,1);
      for iptl=1:nptiles
        mhdI = mhdPtiles(iptl);
        adraw = mhdI*siga;
        bdraw = mhdI*sigb;
        h(iptl) = drawellipse(xc+ifo.mean(1),yc+ifo.mean(2),...
          theta,adraw,bdraw,'color',color,'linewidth',lineWidth);
  
%         foo = invcov*xy';
%         foo = sum(xy'.*foo,1);
%         ninside = nnz(foo<=mhdthis^2);
%         nout = nnz(foo>mhdthis^2);
%         fprintf(1,'%d/%d in/out, %.3f\n',ninside,nout,ninside/(ninside+nout));
      end
    end
    function ifo = gaussianFit(xy)
      % xy: [nx2] points
      %
      % compute mean, cov, invert cov, sqrt(det(cov))
      
      tfnan = any(isnan(xy),2);
      nnan = nnz(tfnan);
      if nnan>0
        warningNoTrace('Removing %d rows containing NaNs.\n',nnan);
        xy = xy(~tfnan,:);
      end      
      
      assert(size(xy,2)==2);
      ifo.mean = mean(xy,1);
      ifo.cov = cov(xy);
      %fprintf(1,'xy mean (expect cntred): %s\n',mat2str(ifo.mean));
      ifo.f = GTPlot.make2DGaussian(ifo.mean,ifo.cov);
      
      ifo.xy = xy;
      [ifo.fxy,ifo.mhd2xy] = ifo.f(xy);
      
      assert(iscolumn(ifo.fxy));
    end
    function f = make2DGaussian(mu,cov)
      invcov = inv(cov);
      sqrtdet = sqrt(det(cov));
      f = @nst;
      function [p,mhd2] = nst(x)
        % x: [nx2]
        %
        % p: [nx1]
        % mhd2: [nx1] maha dist^2
        
        %validateattributes(x,{'numeric'},{'vector' 'numel' 2});
        assert(size(x,2)==2);
        xbar = x-mu; % bsx
        xbar = xbar'; % [2xn]   
        exparg = sum(xbar.*(invcov*xbar),1);
        exparg = exparg';
        mhd2 = exparg;
        p = 1/(2*pi*sqrtdet) * exp(-0.5*exparg);
      end
    end
    
  end
  
end
