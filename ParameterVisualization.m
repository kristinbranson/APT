classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  methods (Abstract)
    
    % Called when a property is selected for consideration, eg when a user
    % clicks a row in the parameter tree
    %
    % hAx: scalar axes. Concrete ParameterVisualizations draw here as
    %   desired.
    % lObj: Labeler obj
    % sPrm: current *NEW-STYLE* params in UI/PropertyTable. All
    %   ParameterSetup/ParameterVisualization code works with parameters in
    %   "new-style" parameter space, as this is how params are presented to
    %   the user.
    %   NOTE: Besides being new-style vs old-style, sPrm in general will
    %   differ from lObj.trackGetParams(), as the PropertyTable may be in 
    %   a modified/edited state and these changes are not written to the 
    %   Labeler until the user clicks Apply.
    init(obj,hAx,lObj,propFullName,sPrm)

    % All args are as in init(...); sPrm contains the latest/updated
    % parameters.
    update(obj,hAx,lObj,propFullName,sPrm)
    
    % Called "on the fly" when a user sets/selects a new property value.
    % The difference between this and update() is that here, the value val
    % may be newer than sPrm.
    updateNewVal(obj,hAx,lObj,propFullName,sPrm,val)
    
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