classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  properties 
    
    axPos = [0.0473118279569892 0.0493358633776091 0.911827956989247 0.929791271347249];
    
  end
  
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
    %   differ from lObj.trackGetTrainingParams(), as the PropertyTable may be in 
    %   a modified/edited state and these changes are not written to the 
    %   Labeler until the user clicks Apply.
    propSelected(obj,hAx,lObj,propFullName,sPrm)
    
    % Called when a property is no longer selected. For cleanup purposes
    propUnselected(obj)

    init(obj,hAx,lObj,propFullName,sPrm)

    % All args are as in init(...); sPrm contains the latest/updated
    % parameters.
    propUpdated(obj,hAx,lObj,propFullName,sPrm)
    
    % Called "on the fly" when a user sets/selects a new property value.
    % The difference between this and update() is that here, the value val
    % may be newer than sPrm.
    propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,val)
    
  end
  
  methods (Static) % Utilities for subclasses
    
    function [paramVizClsname,paramVizID] = parseParamVizSpec(pvSpec)
      toks = regexp(pvSpec,'#','split');
      ntok = numel(toks);
      switch ntok
        case 1
          paramVizClsname = toks{1};
          paramVizID = '';
        case 2
          paramVizClsname = toks{1};
          paramVizID = toks{2};
        otherwise
          error('Invalid ParameterVisualization specification: %s',pgp.ParamViz);
      end
    end
    
    function grayOutAxes(ax,str)
      if nargin<2
        str = '';
      end
      hFig = ancestor(ax,'figure');
      cla(ax);
      ax.Color = hFig.Color;
      title(ax,'');
      ax.XTick = [];
      ax.YTick = [];
      
      if ~isempty(str)
        lims = axis(ax);
        lims = double(lims); % #292 strange err lims can be returned as singles upsetting text()
        xc = (lims(1)+lims(2))/2;
        yc = (lims(3)+lims(4))/2;
        text(xc,yc,str,'horizontalalignment','center','parent',ax);
      end
    end
    
    function setBusy(hAx,str)
      
      if nargin < 2,
        str = 'Updating visualization. Please wait...';
      end
      xlabel(hAx,str);
      set(hAx,'XColor','m');
      drawnow;
      
    end
    
    function setReady(hAx)
      
      xlabel(hAx,sprintf('Visualization updated at %s',datestr(now)));
      set(hAx,'XColor','k');
      drawnow;
      
    end      

    function s = modernizePropName(s)
      if ~startsWith(s,'ROOT.'),
        s = ['ROOT.',s];
      end
    end

    function v = getParamValue(sPrm,fullPath)
      if isstruct(sPrm)
        fns = strsplit(fullPath,'.');
        v = sPrm;
        for i = 1:numel(fns),
          v = v.(fns{i});
        end
      elseif isa(sPrm,'TreeNode'),
        v = sPrm.findnode(fullPath);
        if isempty(v.Children),
          v = v.Data.Value;
        else
          v = v.structize();
          fns = fieldnames(v);
          assert(numel(fns)==1);
          v = v.(fns{1});
        end
      end

    end

  end
  
end