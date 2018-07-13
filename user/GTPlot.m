classdef GTPlot
  methods (Static)
    
    function [hFig,hAxs] = bullseyePtiles(err,I,xyLbl,varargin)
      % Bullseyes overlaid on images
      %
      % I: [nviews] cell array of images to use
      % xyLbl: [npt x 2 x nview] GT labels where bullseyes will be centered
      % err: [n x npts x nviews x nsets] Error array. 
      %  - n is the number of rows/frames
      %  - npts is the number of landmarks
      %  - nviews can be 1 for single-view data
      %  - nsets can be 1 for eg a single XV run, or greater than 1 when 
      %  eg a parameter is titrated it is desired to compare multiple runs.
      %
      % hFig: figure handle
      % hAxs: [nviews x nsets] axes handles
      
      [ptiles,hFig,xyLblPlotArgs,setTitles,ptileCmap] = ...
        myparse(varargin,...
        'ptiles',[50 75 90 95 97.5 99 99.5],...
        'hFig',[],...
        'xyLblPlotArgs',{'m+'},...
        'setTitles',[],...
        'ptileCmap','jet');
      
      [n,npts,nviews,nsets] = size(err);
      assert(isvector(I) && iscell(I) && numel(I)==nviews);
      szassert(xyLbl,[npts,2,nviews]);
      if isempty(setTitles)
        setTitles = arrayfun(@(x)sprintf('Set %d',x),1:nsets,'uni',0);
      end
      assert(iscellstr(setTitles) && numel(setTitles)==nsets);
           
      if isempty(hFig)
        hFig = figure();
        set(hFig,'Color',[1 1 1]);
      else
        figure(hFig);
        clf
      end
      
      nptiles = numel(ptiles);
      err_prctiles = nan(nptiles,npts,nviews,nsets);
      for l=1:npts
        for v=1:nviews
          for k=1:nsets
            err_prctiles(:,l,v,k) = prctile(err(:,l,v,k),ptiles);
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
          plot(ax,xyL(:,1),xyL(:,2),xyLblPlotArgs{:});
          if viewi==1
            tstr = setTitles{k};
            if k==1
              tstr = sprintf('N=%d. %s',n,tstr);
            end
            title(ax,tstr,'fontweight','bold','fontsize',22);
          end
          
          for p = 1:nptiles
            for l = 1:npts
              rad = err_prctiles(p,l,viewi,k);
              h(p) = drawellipse(xyL(l,1),xyL(l,2),0,rad,rad,...
                'Color',colors(p,:),'Parent',ax,'linewidth',1);
            end
          end
        end
      end
            
      legends = cell(1,nptiles);
      for p = 1:nptiles
        legends{p} = sprintf('%sth %%ile',num2str(ptiles(p)));
      end
      hl = legend(h,legends);
      set(hl,'Color','k','TextColor','w','EdgeColor','w');
      truesize(hFig);            
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
      % err: see bullseyePtiles
      %
      % hFig: figure handle
      % hAxs: [nviews x nsets] axes handles
      
      [ptiles,hFig,lineArgs,setNames,axisArgs] = ...
        myparse(varargin,...
        'ptiles',[50 75 90 95 97.5 99 99.5],...
        'hFig',[],...
        'lineArgs',{'m+'},...
        'setNames',[],...
        'axisArgs',{'XTicklabelRotation',45,'FontSize' 16}...
        );
      
      [n,npts,nviews,nsets] = size(err);     
     
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
      
      hAxs = createsubplots(nviews,npts+1,[.05 0;.12 .12]);
      hAxs = reshape(hAxs,nviews,npts+1);
      for ivw=1:nviews
        for ipt=[1:npts inf]
          if ~isinf(ipt)
            % normal branch
            errs = squeeze(err(:,ipt,ivw,:)); % nxnsets
            y = prctile(errs,ptiles); % [nptlsxnsets]
            ax = hAxs(ivw,ipt);
            tstr = sprintf('vw%d pt%d',ivw,ipt);
          else
            errs = squeeze(sum(err(:,:,ivw,:),2)/npts); % [nxnsets]
            y = prctile(errs,ptiles); % [nptlsxnsets]
            ax = hAxs(ivw,npts+1);
            tstr = sprintf('vw%d, mean allpts',ivw);
          end
          axes(ax);
          tfPlot1 = ivw==1 && ipt==1;
%           if tfPlot1
%             tstr = ['XV err vs CropType: ' tstr];
%           end
          
          args = {...
            'YGrid' 'on' 'XGrid' 'on' 'XLim' [0 nsets+1] 'XTick' 1:nsets ...
            'XTickLabel',setNames};
          args = [args axisArgs];
            
          x = 1:nsets; % croptypes
          h = plot(x,y','.-','markersize',20);
          set(ax,args{:});
          hold(ax,'on');
          ax.ColorOrderIndex = 1;
          
          if tfPlot1
            tstr = sprintf('N=%d. %s',n,tstr);
            legstrs = strcat(numarr2trimcellstr(ptiles'),'%');
            hLeg = legend(h,legstrs);
            hLeg.FontSize = 10;
            %xlabel('Crop type','fontweight','normal','fontsize',14);
            
            ystr = sprintf('raw err (px)');
            ylabel(ystr,'fontweight','normal','fontsize',14);
          else
            set(ax,'XTickLabel',[]);
          end
          title(tstr,'fontweight','bold','fontsize',16);
          if ipt==1
          else
            set(ax,'YTickLabel',[]);
          end
        end
      end
      linkaxes(hAxs(1,:));
      linkaxes(hAxs(2,:));
%       ylim(axs(1,1),[0 50]);
%       ylim(axs(2,1),[0 80]);
      %linkaxes(axs(2,:),'y');
      % ylim(axs(2,1),[0 20]);
      
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
      hAxs = subplots(nptiles,npts,[.12 0;.12 0]);
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
            yl(2) = 2*y(iptile,1); 
            ylim(ax,yl);
          end
          set(ax,'YTick',ax.YTick(1:end-1),'YTickLabel',ax.YTickLabel(1:end-1));
        end
      end
      
      for iptile=1:nptiles
        linkaxes(hAxs(iptile,:));
      end
    end
  end
end
