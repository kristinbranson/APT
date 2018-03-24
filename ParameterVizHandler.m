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
    hAx % scalar visualization axis
    prop2ParamViz % containers.Map. key: prop object. val: ParameterVisualziation obj
  end
  
  methods
    
    function obj = ParameterVizHandler(labelerObj,ax)
      obj.lObj = labelerObj;
      obj.hAx = ax;
      obj.prop2ParamViz = containers.Map();
    end
    
    function init(obj)
      ax = obj.hAx;
      ax.Visible = 'off';
      ParameterVisualization.grayOutAxes(obj.hAx);
    end
    
    function addProp(obj,prop,pvObj)
      key = char(prop.toString);
      obj.prop2ParamViz(key) = pvObj;
      fprintf('Added prop: %s\n',key);
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
        val0 = prop.getValue();
        lblObj = obj.lObj;
        sPrm = lblObj.trackGetParams();
        pvObj.init(obj.hAx,lblObj,sPrm,val0);
      else
        ParameterVisualization.grayOutAxes(obj.hAx,...
          'No visualization available.');
      end
    end
    
    function propUpdated(obj,pvObj,spinnerEvt)
      % Called when prop's spinner is clicked
      % 
      % pvObj: prop->pvObj
      
      pvObj.update(obj.hAx,obj.lObj,struct(),spinnerEvt.spinnerValue);
    end
    
  end
end