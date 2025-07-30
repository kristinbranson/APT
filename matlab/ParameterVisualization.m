classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  properties
    hTile
    hAx
    lObj
    prm
    propFullName
    is_ma = false;
    is2stage = false;
    is_ma_net = false;
    stage = 1;
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
    
    % parameters have changed, update plot
    update(obj)
        
  end

  methods

    % initialize the plot and properties
    function init(obj,hTile,lObj,propFullName,prm,varargin)
      obj.hTile = hTile;
      obj.hAx = gobjects(1,0);
      obj.lObj = lObj;
      obj.prm = prm;
      obj.propFullName = propFullName;
    end

    % For cleanup purposes
    function clear(obj)
      cla(obj.hAx);    
      for i = 1:numel(obj.hAx),
        obj.hAx(i).Title.String = '';
        obj.hAx(i).XLabel.String = '';
        obj.hAx(i).YLabel.String = '';
        obj.hAx(i).ZLabel.String = '';
        delete(obj.hAx(i).Legend);
      end
    end

    function setStage(obj)

      [~,idx] = regexp(obj.propFullName,'\.DeepTrack','once');
      if isempty(idx),
        return;
      end
      propPrefix = obj.propFullName(1:idx);

      obj.is_ma = obj.lObj.maIsMA;
      obj.is2stage = obj.lObj.trackerIsTwoStage;
      obj.is_ma_net = false;
      obj.stage = 1;

      if obj.is_ma,
        if obj.is2stage && startsWith(propPrefix,'ROOT.DeepTrack'),
          obj.stage = 2;
        else
          obj.is_ma_net = true;
        end
      end
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

    function hAx = getFirstAx(obj)
      hAx = obj.hAx(find(ishandle(obj.hAx),1));
    end

    function grayOutAxes(obj,str)
      obj.clear();
      if isempty(obj.hAx) || all(~ishandle(obj.hAx)),
        return;
      end
      [obj.hAx(ishandle(obj.hAx)).Visible] = 'off';
      if nargin >= 2 && ~isempty(str)
        hti = title(obj.getFirstAx(),str);
        hti.Visible = 'on';
      end
    end
    
    function setBusy(obj,str)
      
      if isempty(obj.hAx) || all(~ishandle(obj.hAx)),
        return;
      end
      if nargin < 2,
        str = 'Updating visualization. Please wait...';
      end
      hax = obj.getFirstAx();
      xlabel(hax,str);
      set(hax,'XColor','m');
      drawnow;
      
    end
    
    function setReady(obj)
      
      if isempty(obj.hAx) || all(~ishandle(obj.hAx)),
        return;
      end
      hax = obj.getFirstAx();
      xlabel(hax,sprintf('Visualization updated at %s',datestr(now)));
      set(hax,'XColor','k');
      drawnow;
      
    end      

  end
  
end