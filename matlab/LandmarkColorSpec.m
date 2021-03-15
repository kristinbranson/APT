classdef LandmarkColorSpec < matlab.mixin.Copyable
  % Color Specification
  %
  % * Colors may be spec'd "automatically" via either colormapname, 
  % or "manually" by direct spec of colors
  % 
  % In Labeler projects and eg lObj.labelPointsPlotInfo,
  % .prePointsPlotInfo, etc, colors are stored only via .ColorMapName
  % and .Colors. The "concretized" colors stored in .Colors are what are
  % actually used thruout APT. The ColorMapName is used only to guide eg
  % the LandmarkColors UI, eg the concrete .Colors may have been derived
  % from the .ColorMapName -- using a ColorMap as a starting point and then 
  % brightening/dimming, or manually changing some subset of colors, can be 
  % it can be useful workflow.
  %
  % Since APT projects do not store eg the brightness, it can only be 
  % guessed as to i) whether  .Colors were generated directly from 
  % .ColorMapName, and ii) if so, what brightness was used. This is a
  % little subtle but seems fine for now.
  
  properties (Constant)
    % doesn't appear to be programmatically avail, inexplicably
    CMAPNAMES = {
      'jet'
      'parula'
      'hsv'
      'lines'
      'autumn'
      'spring'
      'winter'
      'summer'
      'hot'
      'cool'
      };
  end
  properties
    landmarkSetType % scalar LandmarkSetType
    
    nphyspts % number of physical pts, as pts are identified across views
    colormapname % eg 'jet'
    colormap % colormapname, instantiated per npts and brightness
    brightness % scalar
    
    % could be manually adjusted per-pt, or set manually whole cloth; can 
    % also be auto-colors if ~tfmanual (yes this is a little confused)
    colors 
    tfmanual % if false, use .colormap; else, use .colors 
  end
  methods
    function obj = LandmarkColorSpec(lsettype,nphyspts,pointsPlotInfo)
      % name: arbitrary id
      % pointsPlotInfo: eg lObj.labelPointsPlotInfo. Must have fields
      %   .ColorMapName, .Colors
      
      obj.landmarkSetType = lsettype;
      obj.nphyspts = nphyspts;
      
      ppi = pointsPlotInfo;
      cmapname = ppi.ColorMapName;
      tf = strcmp(cmapname,LandmarkColorSpec.CMAPNAMES);
      if ~any(tf)
        warningNoTrace('Unknown colormap name %s, using default ''jet''',cmapname);
        cmapname = 'jet';
      end      
      obj.colormapname = cmapname;
      obj.colors = ppi.Colors;
      
      obj.guessBrightness();
      obj.updateColormap();      
    end
    function copyColorState(obj,obj2)
      % Copy color-state from obj2 onto obj
      FLDS = {'colormapname' 'colormap' 'brightness' 'colors' 'tfmanual'};
      for f=FLDS,f=f{1};
        obj.(f) = obj2.(f);
      end
    end
    function guessBrightness(obj)
      % sets .brightness and .tfmanual 
      
      obj.brightness = .5;
      obj.tfmanual = true;
      
      colormap = feval(obj.colormapname,obj.nphyspts); %#ok<*PROP>
      ratio = obj.colors./colormap;
      ratio(isnan(ratio)) = [];
      
      % give up
      if std(ratio(:)) > .02,
        return;
      end
      
      meanratio = mean(ratio);
      if meanratio > 1,
        v = max(0,min(1,regress(obj.colors(:)-colormap(:),1-colormap(:))));
        newcolormap = (1-v)*colormap + v;
        err = abs(obj.colors(:)-newcolormap(:));
        if max(err) > .02,
          return;
        end
        obj.brightness = .5 + v/2;
      elseif meanratio < 1,
        v = max(0,min(1,regress(obj.colors(:),colormap(:))));
        newcolormap = v*colormap;
        err = abs(obj.colors(:)-newcolormap(:));
        if max(err) > .02,
          return;
        end
        obj.brightness = v/2;
      else
        obj.brightness = .5;
      end
      
      obj.tfmanual = false;
    end
    function updateColormap(obj)
      obj.colormap = feval(obj.colormapname,obj.nphyspts);
      if obj.brightness > .5,
        v = min(1,max(0,(obj.brightness-.5)*2));
        obj.colormap = obj.colormap*(1-v)+v;
      elseif obj.brightness < .5,
        v = min(1,max(0,obj.brightness*2));
        obj.colormap = obj.colormap*v;
      end
    end
    function setColormapName(obj,cmapname)
      obj.colormapname = cmapname;
      obj.updateColormap();
      obj.tfmanual = false;
    end
    function setBrightness(obj,v)
      obj.brightness = v;
      obj.updateColormap();
      obj.tfmanual = false;
    end
    function setColorManual(obj,iphyspt,clr)
      obj.colors(iphyspt,:) = clr;
      obj.tfmanual = true;
    end
    function setManualColorsToColormap(obj)
      obj.colors = obj.colormap;
    end
    function setTFManual(obj,tf)
      obj.tfmanual = tf;
    end
    function setManualColorsToColormapIfNec(obj)
      % obj: can be array
      for i=1:numel(obj)
        if ~obj(i).tfmanual
          obj(i).setManualColorsToColormap();
        end
      end
    end
      
  end
  
end