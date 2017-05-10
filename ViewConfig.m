classdef ViewConfig
  
  properties (Constant)
    GAMMA_CORRECT_CMAP_LEN = 256;
  end
  
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
        
    function tfAxLimSpecifiedInCfg = setCfgOnViews(viewCfg,hFig,hAx,hIm,hAxPrev)
      % viewCfg: currently just a struct array
      %
      % This not only sets the stuff in viewCfg, but also resets some stuff
      % to "default"/auto if viewCfg doesn't say anything (has empty props)
      
      nview = numel(hFig);
      assert(isequal(nview,numel(hAx),numel(viewCfg)));
      tfAxLimSpecifiedInCfg = false(nview,1);
      
      for iView = 1:nview
        vCfg = viewCfg(iView);
        ax = hAx(iView);
        
        fpos = vCfg.FigurePos;
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
        
        axlim = vCfg.AxisLim;
        tf = structfun(@isscalar,axlim);
        if all(tf)
          axlim = [axlim.xmin axlim.xmax axlim.ymin axlim.ymax];
          axis(ax,axlim);
          tfAxLimSpecifiedInCfg(iView) = true;
        else
          if any(tf)
            warning('LabelerGUI:axLim',...
              'Ignoring invalid configuration setting: axis limits for axis %d.',iView);
          end
          
          axis(ax,'image'); % "auto"/default      
        end
        
        hlpAxDir(ax,'XDir',vCfg.XDir);
        hlpAxDir(ax,'YDir',vCfg.YDir);
        if iView==1
          hlpAxDir(hAxPrev,'XDir',vCfg.XDir);
          hlpAxDir(hAxPrev,'YDir',vCfg.YDir);
        end
        
        clim = vCfg.CLim;
        if isempty(clim.Min) && isempty(clim.Max)
          caxis(ax,'auto');
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
          ax.CLim = climset;
          if iView==1
            hAxPrev.CLim = climset;
          end
        end
        
        gam = vCfg.Gamma;
        if ~isempty(gam)
          ViewConfig.applyGammaCorrection(hIm,hAx,hAxPrev,iView,gam);
        else
          cm = gray(ViewConfig.GAMMA_CORRECT_CMAP_LEN);
          colormap(ax,cm);
          ax.UserData.gamma = [];
        end

        ax.XAxisLocation = 'top';
        ax.XColor = vCfg.AxColor;
        ax.YColor = vCfg.AxColor;
        ax.Box = 'on';
        ax.FontUnits = 'pixels';
        ax.FontSize = vCfg.AxFontSize;
        if vCfg.ShowAxTicks
          ax.XTickMode = 'auto';
          ax.YTickMode = 'auto';          
        else
          ax.XTickLabel = [];
          ax.YTickLabel = [];
        end
        if vCfg.ShowGrid
          grid(ax,'on');
        else
          grid(ax,'off');
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
            cl = ax.CLim;
            viewCfg(i).CLim = struct('Min',cl(1),'Max',cl(2));
          otherwise
            viewCfg(i).CLim = struct('Min',[],'Max',[]);
        end
        fpos = fg.Position;
        viewCfg(i).FigurePos = struct(...
          'left',fpos(1),...
          'bottom',fpos(2),...
          'width',fpos(3),...
          'height',fpos(4));
        if strcmp(ax.XLimMode,'manual') && strcmp(ax.YLimMode,'manual') % right now require both
          xl = ax.XLim;
          yl = ax.YLim;
          viewCfg(i).AxisLim.xmin = xl(1);
          viewCfg(i).AxisLim.xmax = xl(2);
          viewCfg(i).AxisLim.ymin = yl(1);
          viewCfg(i).AxisLim.ymax = yl(2);
        end
        
        if isfield(ax.UserData,'gamma') && ~isempty(ax.UserData.gamma)
          viewCfg(i).Gamma = ax.UserData.gamma;
        end
        
        viewCfg(i).ShowAxTicks = ~isempty(ax.XTickLabel);
        viewCfg(i).ShowGrid = strcmp(ax.XGrid,'on');
      end
    end
    
    function applyGammaCorrection(hIms,hAxs,hAxPrev,iAxApply,gamma)
      assert(numel(hIms)==numel(hAxs));

      for iAx = iAxApply(:)'
        im = hIms(iAx);
        if size(im.CData,3)~=1
          error('Labeler:gamma',...
            'Gamma correction currently only supported for grayscale/intensity images.');
        end
      end
      
      m0 = gray(ViewConfig.GAMMA_CORRECT_CMAP_LEN);
      m1 = imadjust(m0,[],[],gamma);

      for iAx = iAxApply(:)'
        colormap(hAxs(iAx),m1);
        hAxs(iAx).UserData.gamma = gamma; % store gamma value that was applied
        if iAx==1 % axes_curr
          colormap(hAxPrev,m1);
        end
      end
    end

    function movInvert = getMovieInvert(viewCfg)
      nview = numel(viewCfg);
      movInvert = false(1,nview);
      for i=1:nview
        if ~isempty(viewCfg(i).InvertMovie)
          movInvert(i) = logical(viewCfg(i).InvertMovie);
        end
      end
    end
  end
  
end

function hlpAxDir(ax,prop,val)
if isempty(val)
  % none, leave as-is, no default/"auto" value
elseif any(strcmp(val,{'n' 'r' 'normal' 'reverse'}))
  ax.(prop) = val;
else
  warning('LabelerGUI:axDir',...
    'Ignoring invalid configuration setting ''%s'' for axis.%s.',...
    val,prop);
end
end