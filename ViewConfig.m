classdef ViewConfig  
  
  % At new-project-from-config time (cbkNewProject), all figs/axes are
  % created and setViewsToViewCfg is called to set things up accordingly.
  %
  % At new-movie time, nothing is reset to configuration; all settings remain
  % as-the-user-has-left. Except, if the axis limits were not specified, then
  % the axis is set to 'image' for a tight fit etc.
  %
  % There will be a UI option to "reset views to config" which will set all
  % view-stuff to configs at any time. This lets the user play with zoom,
  % adjusting brightness, flip etc and be able to restore.
  %
  % At save-time, all view stuff is saved AS IN THE CURRENT STATE! So make
  % sure to reset the view if you messed everything up.
  
  methods (Static)
    
    function tfAxLimSpecifiedInCfg = setCfgOnViews(viewCfg,hFig,hAx,hAxPrev)
      % viewCfg: currently just a struct array
      
      nview = numel(hFig);
      assert(isequal(nview,numel(hAx),numel(viewCfg)));
      tfAxLimSpecifiedInCfg = false(nview,1);
      
      for iView = 1:nview
        fpos = viewCfg(iView).FigurePos;
        tf = structfun(@isscalar,fpos);
        if all(tf)
          fpos = [fpos.left fpos.bottom fpos.width fpos.height];
          hFig(iView).Position = fpos;
        elseif any(tf)
          warning('LabelerGUI:figPos',...
            'Ignoring invalid configuration setting: position for figure %d.',iView);
        else
          % all fpos elements are empty; no-op
        end
        
        axlim = viewCfg(iView).AxisLim;
        tf = structfun(@isscalar,axlim);
        if all(tf)
          axlim = [axlim.xmin axlim.xmax axlim.ymin axlim.ymax];
          axis(hAx(iView),axlim);
          tfAxLimSpecifiedInCfg(iView) = true;
        elseif any(tf)
          warning('LabelerGUI:axLim',...
            'Ignoring invalid configuration setting: axis limits for axis %d.',iView);
        else
          % all axis elements are empty; no-op
        end
        
        hlpAxDir(hAx(iView),'XDir',viewCfg(iView).XDir);
        hlpAxDir(hAx(iView),'YDir',viewCfg(iView).YDir);
        
        clim = viewCfg(iView).CLim;
        if isempty(clim.Min) && isempty(clim.Max)
          caxis(hAx(iView),'auto');
          if iView==1
            caxis(hAxPrev,'auto');
          end
        else
          climset = [0 inf]; % using inf seems to work as "auto-scale-maximum"
          if ~isempty(clim.Min)
            climset(1) = clim.Min;
          end
          if ~isempty(clim.Max)
            climset(2) = clim.Max;
          end
          hAx(iView).CLim = climset;
          if iView==1
            hAxPrev.CLim = climset;
          end
        end
      end      
    end
    
    function viewCfg = readCfgOffViews(hFig,hAx,hAxPrev)
      % viewCfg: currently just a struct array
      
      assert(numel(hFig)==numel(hAx));
      nview = numel(hFig);
      
      s = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      viewCfg = repmat(s.View,nview,1);
      
      for i=1:nview
        fg = hFig(i);
        ax = hAx(i);
        
        viewCfg(i).XDir = ax.XDir;
        viewCfg(i).YDir = ax.YDir;
        switch ax.CLimMode
          case 'manual'
            viewCfg(i).CLim = ax.CLim;
          otherwise
            viewCfg(i).CLim = [];
        end
        viewCfg(i).FigurePos = fg.Position;
        if strcmp(ax.XLimMode,'manual') && strcmp(ax.YLimMode,'manual') % right now require both
          xl = ax.XLim;
          yl = ax.YLim;
          viewCfg(i).AxisLim.xmin = xl(1);
          viewCfg(i).AxisLim.xmax = xl(2);
          viewCfg(i).AxisLim.ymin = yl(1);
          viewCfg(i).AxisLim.ymax = yl(2);          
        end
      end
    end
    
  end
  
end