classdef SegmentedLine < handle
% A segmented line is a matlab.graphics.primitive.Line that can be toggled
% on or off at integer x-locations.
  
  properties
    hAx
    hLine
    
    xLims
    yLoc
  end
  
  methods
    
    function obj = SegmentedLine(ax,tag)
      % ax: parent axis
          
      if nargin <= 1,
        tag = 'SegmentedLine';
      end
        
      assert(isa(ax,'matlab.graphics.axis.Axes'));
      hL = line('XData',nan,'YData',nan,'Parent',ax,'Tag',tag);
      
      obj.hAx = ax;
      obj.hLine = hL;
    end
    
    function init(obj,xlim,yloc,onProps)
      % xlim: [2] line x-limits
      % yloc: [1] line y-loc
      % onProps: line P-V structure for on appearance
      
      assert(numel(xlim)==2);
      assert(isscalar(yloc));
      assert(isstruct(onProps));

      x = [xlim(1)-0.5:xlim(2)-0.5;xlim(1)+0.5:xlim(2)+0.5];
      x(end+1,:) = nan;
      y = nan(size(x));
      set(obj.hLine,'XData',x(:),'YData',y(:));
      set(obj.hLine,onProps);
      
      obj.xLims = xlim;
      obj.yLoc = yloc;     
    end
    
    function delete(obj)
      if isvalid(obj.hLine)
        delete(obj.hLine);
      end
      obj.hLine = [];
    end
    
    function setOnAtOnly(obj,x)
      obj.setOnOffAt(x,true);
      xall = obj.xLims(1):obj.xLims(2);
      xcomp = setdiff(xall,x);
      obj.setOnOffAt(xcomp,false);
    end
    
      
    function setOnOffAt(obj,x,tf)
      % Set segment on/off at location(s) x
      %
      % x: positive integer vector of coords/locs
      
      if isempty(x)
        return;
      end
      x = x(:);
        
      assert(all(obj.xLims(1)<=x & x<=obj.xLims(2)));
      
      xx = (x-1)*3;
      yDataIdx = [xx+1;xx+2];
      if tf
        obj.hLine.YData(yDataIdx) = obj.yLoc;
      else
        obj.hLine.YData(yDataIdx) = nan;
      end
    end
    
    function setVisible(obj,tf)
      set(obj.hLine,'Visible',onIff(tf));
    end
    
  end
  
end