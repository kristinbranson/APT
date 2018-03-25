classdef ParameterVizHandler < handle
  % Manages parameter visualization. Connects Labeler, ParameterSetup, and
  % ParameterVisualizations
  %
  % - holds concrete ParameterVisualization objects
  % - accesses Labeler to generate default image, get labels etc
  % - Called by ParameterSetup when user selects new prop, updates/edits
  % prop value
  % - Forwards to ParameterVisualizations appropriately
  
  properties
    lObj 
    hFig % scalar fig handle, SetParameter fig. For accessing appdata
    hAx % scalar visualization axis
    prop2ParamViz % containers.Map. key: prop object. val: ParameterVisualization obj
  end
  
  methods
    
    function obj = ParameterVizHandler(labelerObj,hFig,ax)
      obj.lObj = labelerObj;
      obj.hFig = hFig;
      obj.hAx = ax;
      obj.prop2ParamViz = containers.Map();
    end
    
    function delete(obj)
      obj.lObj = [];
      obj.hFig = [];
      obj.hAx = [];
      delete(obj.prop2ParamViz);
      obj.prop2ParamViz = [];
    end
    
    function init(obj)
      ax = obj.hAx;
      ax.Visible = 'off';
      ParameterVisualization.grayOutAxes(obj.hAx);
    end
    
    function addProp(obj,prop,pvObj)
      key = char(prop.toString);
      obj.prop2ParamViz(key) = pvObj;
%       fprintf('Added prop: %s\n',key);
    end
    
    function [tf,pvObj] = isprop(obj,prop)
      key = char(prop.toString);
      m = obj.prop2ParamViz;
      tf = m.isKey(key);
      if tf
        pvObj = m(key);
      else
        pvObj = [];
      end
    end
    
    function propSelected(obj,prop)
      % Called when a prop is initially selected
      % 
      % prop: grid java prop
      
      [tf,pvObj] = obj.isprop(prop);
      if tf
%         val0 = prop.getValue();
        lblObj = obj.lObj;
        sPrm = obj.getCurrentParamsInTree();
        fprintf(1,'PVH calling init\n');
        pvObj.init(obj.hAx,lblObj,char(prop.getFullName()),sPrm);
      else
        ParameterVisualization.grayOutAxes(obj.hAx,...
          'No visualization available.');
      end
    end
    
    function propUpdatedGeneral(obj,prop)
      [tf,pvObj] = obj.isprop(prop);
      if tf
        fprintf(1,'PVH calling update\n');
        sPrm = obj.getCurrentParamsInTree();        
        pvObj.update(obj.hAx,obj.lObj,char(prop.getFullName()),sPrm);
      end
    end
    
    function propUpdatedSpinner(obj,prop,pvObj,spinnerEvt)
      % Called when prop's spinner is clicked. propertyTable.appData.mirror
      % has not been updated yet
      % 
      % pvObj: prop->pvObj
      
      fprintf(1,'PVH calling updateNewVal\n');
      sPrm = obj.getCurrentParamsInTree(); % sPrm outdated relative to spinnerEvt.spinnerValue;
      pvObj.updateNewVal(obj.hAx,obj.lObj,char(prop.getFullName()),...
        sPrm,spinnerEvt.spinnerValue);
    end
    
    function sPrm = getCurrentParamsInTree(obj)
      % sPrm: NEW-STYLE Parameters. See notes in ParameterVisualization.
      
      mirror = getappdata(obj.hFig,'mirror');
      rootnode = getappdata(obj.hFig,'rootnode');
      % AL: Some contorting here with rootnode; see ParameterSetup.
      % There's prob a better way
      rootnode.Children = mirror;
      sPrm = rootnode.structize();
      rootnode.Children = [];
    end
    
  end
end