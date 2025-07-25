classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  properties
    hAx
    lObj
    prm
    propFullName
  end

  methods (Abstract)
    
    % Called when a property is visualized
    % hAx: scalar axes. Concrete ParameterVisualizations draw here as
    %   desired.
    % lObj: Labeler obj
    % prm: current params, TreeNode object
    %   NOTE: Besides being new-style vs old-style, sPrm in general will
    %   differ from lObj.trackGetTrainingParams(), as the PropertyTable may be in 
    %   a modified/edited state and these changes are not written to the 
    %   Labeler until the user clicks Apply.
    
    % For cleanup purposes
    clear(obj)

    % parameters have changed, update plot
    update(obj)
        
  end

  methods

    % initialize the plot and properties
    function init(obj,hAx,lObj,propFullName,prm)
      obj.hAx = hAx;
      obj.lObj = lObj;
      obj.prm = prm;
      obj.propFullName = propFullName;
    end

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

  methods 
    
    function hs = getAxesAndFriendsHandles(obj)

      hs = [];
      if isempty(obj.hAx) || ~ishandle(obj.hAx),
        return
      end
      hs = [obj.hAx.Title,obj.hAx.XLabel,obj.hAx.YLabel,obj.hAx.ZLabel,obj.hAx.Legend];
      hs = hs(ishandles(hs));
      
    end

    function grayOutAxes(obj,str)
      if isempty(obj.hAx) || ~ishandle(obj.hAx),
        return;
      end
      obj.hAx.Visible = 'off';
      hs = obj.getAxesAndFriendsHandles();
      [hs.Visible] = deal('off');

      if nargin >= 2 && ~isempty(str)
        hti = title(obj.hAx,str);
        hti.Visible = 'on';
      end
    end
    
    function setBusy(obj,str)
      
      if isempty(obj.hAx) || ~ishandle(obj.hAx),
        return;
      end
      if nargin < 2,
        str = 'Updating visualization. Please wait...';
      end
      xlabel(obj.hAx,str);
      set(obj.hAx,'XColor','m');
      drawnow;
      
    end
    
    function setReady(obj)
      
      if isempty(obj.hAx) || ~ishandle(obj.hAx),
        return;
      end
      xlabel(obj.hAx,sprintf('Visualization updated at %s',datestr(now)));
      set(obj.hAx,'XColor','k');
      drawnow;
      
    end      

    function s = modernizePropName(s)
      if ~startsWith(s,'ROOT.'),
        s = ['ROOT.',s];
      end
    end

  end
  
end