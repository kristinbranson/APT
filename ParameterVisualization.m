classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  methods (Abstract)
    
    % Called when a property is selected for consideration, eg when a user
    % clicks a row in the parameter tree
    %
    % hAx: scalar axes. Concrete ParameterVisualizations draw here as
    % desired.
    % imLbl: [nrxnc]. full frame/image for view 1, for some
    % reasonably-selected labeled frame (eg the current frame if possible).
    % pLbl: [nphysptsx2]. labeled points for view 1, same frame as imLbl.
    % movLbl,frmLbl,tgtLbl. movie/frame/target for imLbl/pLbl.
    % lObj
    % sPrm: starting/current params in UI
    init(obj,hAx,lObj,sPrm0,val0)
    
    % Called when a user sets/selects a new property value
    %
    % All args are as in init(...), except sPrm contains the latest/updated
    % parameters.
    update(obj,hAx,lObj,sPrm0,val)
    
  end
  
  methods (Static) % Utilities for subclasses
    
    function grayOutAxes(ax,str)
      if nargin<2
        str = '';
      end
      hFig = ancestor(ax,'figure');
      cla(ax);
      ax.Color = hFig.Color;
      title(ax,'');
      
      if ~isempty(str)
        lims = axis(ax);
        xc = (lims(1)+lims(2))/2;
        yc = (lims(3)+lims(4))/2;
        text(ax,xc,yc,str,'horizontalalignment','center');
      end
    end
        
  end
  
end