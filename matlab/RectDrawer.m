classdef RectDrawer < handle
  % Draw/show rectangles on an axis
  
  properties
    ax
    hRect
  end
  
  methods
    function obj = RectDrawer(hax)
      obj.ax = hax;
      obj.hRect = gobjects(0,1);      
    end
    function delete(obj)
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = [];
      obj.ax = [];
    end
    function initRois(obj)
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = gobjects(0,1);
    end
    function newRoiDraw(obj)
      hax = obj.ax;
      lims = axis(hax);
%       STARTSZFAC = 0.2;
%       wh = STARTSZFAC * (lims(2)-lims(1));
%       ht = STARTSZFAC * (lims(4)-lims(3)); 
%       pos = [lims([1 3]) wh ht];
      try
        h = drawrectangle(hax,'Rotatable',false);
        pos = h.Position;
        h.InteractionsAllowed = 'none';
      catch ME
        warning(getReport(ME));
      end
      if numel(pos) < 4,
        try
          delete(h);
        catch ME
          warning(getReport(ME));
        end
        return;
      end
      if (pos(1)+pos(3))<lims(1) ...
        || (pos(2)+pos(4))<lims(3) ...
        || pos(1)>lims(2) ...
        || pos(2)>lims(4)
        try
          delete(h);
        catch ME
          warning(getReport(ME));
        end
        warningNoTrace('ROI not added: ROI is completely outside the axes limits.');
        return;
      end
      if pos(3)==0 || pos(4)==0
        try
          delete(h);
        catch ME
          warning(getReport(ME));
        end
        warningNoTrace('ROI not added: ROI has zero area.');
        return;
      end
      pos(1:2) = max(lims([1 3]),pos(1:2));
      pos(3:4) = min(lims([2 4]),pos(3:4)+pos(1:2))-pos(1:2);
      h.Position = pos;
%       h = images.roi.Rectangle(obj.ax,'Position',pos,...
%         'InteractionsAllowed','none');
      h.addlistener('DeletingROI',@(s,e)obj.cbkROIDeleted(s,e));
      obj.hRect(end+1,1) = h;
    end
    function setEdit(obj,tf)
      if isempty(obj.hRect)
        return;
      end
      if tf
        val = 'all';
      else
        val = 'none';
      end
      [obj.hRect.InteractionsAllowed] = deal(val);
    end
    function cbkROIDeleted(obj,src,evt)
      tf = obj.hRect==src;
      obj.hRect(tf,:) = [];
    end
    function rmRoi(obj,iRoi)
      delete(obj.hRect(iRoi));
      obj.hRect(iRoi,:) = [];
    end
    function setRois(obj,v)
      % v: [4 x 2 x nroi] vertices; as in image.roi.Rectangle.Vertices
      %
      % This deletes/clears all existing Rois and draws new ones in
      % non-interactive mode.
      
      if isempty(obj.hRect) && isempty(v)
        return;
      end
      
      obj.initRois();
      nroi = size(v,3);
      h = gobjects(nroi,1);
      hax = obj.ax;
      for iroi=1:nroi
        wh = v(3,1,iroi) - v(1,1,iroi);
        ht = v(2,2,iroi) - v(1,2,iroi);
        pos = [v(1,1,iroi) v(1,2,iroi) wh ht];
        h(iroi) = images.roi.Rectangle(hax,'Position',pos,...
          'InteractionsAllowed','none');
        h(iroi).addlistener('DeletingROI',@(s,e)obj.cbkROIDeleted(s,e));
      end
      obj.hRect = h;
    end
    function setShowRois(obj,tf)
      onoff = onIff(tf);
      h = obj.hRect;
      if ~isempty(h)
        [obj.hRect.Visible] = deal(onoff);
      end
    end
    function v = getRoisVerts(obj)
      % v: [4 x 2 x nroi]
      if isempty(obj.hRect)
        v = nan(4,2,0);
      else
        v = cat(3,obj.hRect.Vertices);
      end
    end
  end
    
end
    
  