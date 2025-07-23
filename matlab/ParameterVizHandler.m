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
    cbkToggleParamViz % if true, expand Tracking Params pane to exposeproptoparam viz panel
    
    prop2ParamViz % containers.Map. key: prop object. val: ParameterVisualization obj
    id2ParamViz % containers.Map. key: ID. val: ParameterVisualization obj. Note, not all ParameterVisualizations will have an ID.
    
    paramVizSelected % Either [], or scalar ParameterVisualization associated with currently selected property
    cbkTS = 0;
    fcnPropToKey = [];
  end
  
  methods
    
    function obj = ParameterVizHandler(labelerObj,hFig,ax,cbkTogglePV,varargin)      
      obj.lObj = labelerObj;
      obj.hFig = hFig;
      obj.hAx = ax;
      obj.cbkToggleParamViz = cbkTogglePV;
      
      obj.prop2ParamViz = containers.Map();
      obj.id2ParamViz = containers.Map();
      
      obj.paramVizSelected = [];
      obj.cbkTS = 0;

      [obj.fcnPropToKey] = myparse(varargin,'fcnPropToKey',@(p) char(p.toString));

    end
    
    function delete(obj)
      obj.lObj = [];
      obj.hFig = [];
      obj.hAx = [];
      obj.cbkToggleParamViz = [];
      delete(obj.prop2ParamViz);
      obj.prop2ParamViz = [];
      delete(obj.id2ParamViz);
      obj.id2ParamViz = [];
      obj.paramVizSelected = [];
    end
    
    function init(obj)
      ax = obj.hAx;
      ax.Visible = 'off';
      ParameterVisualization.grayOutAxes(obj.hAx);
    end
    
    function addProp(obj,prop,pvSpec)
      [pvClsname,pvID] = ParameterVisualization.parseParamVizSpec(pvSpec);
      [tf,pvObj] = obj.ispropID(pvID); % pvID could be empty which should return false
      if tf
        obj.addPropExisting(prop,pvObj,pvID);
      else
        try
          pvObj = feval(pvClsname);
          assert(isa(pvObj,'ParameterVisualization'),'''%s'' is not a ParameterVisualization.');
        catch ME
          warningNoTrace('Failed to instantiate ParameterVisualization ''%s'': %s',...
            pvClsname,ME.message);
          pvObj = [];
        end
        if ~isempty(pvObj)
          obj.addPropNew(prop,pvObj,pvID);
        else
          % obj unchanged, warning thrown
        end
      end
    end
    
    function addPropNew(obj,prop,pvObj,pvID)
      % pvID: optional. If non-empty, pvObj is added with ID pvID so can be
      % accessed later.
      
      if ~isempty(pvID) && obj.id2ParamViz.isKey(pvID)
        error('ParameterVisualization with ID ''%s'' has already been added.',pvID);
      end
      
      key = obj.fcnPropToKey(prop);
      obj.prop2ParamViz(key) = pvObj;
      if ~isempty(pvID)
        obj.id2ParamViz(pvID) = pvObj;
      end
    end
    
    function addPropExisting(obj,prop,pvObj,pvID)
      key = obj.fcnPropToKey(prop);
      obj.prop2ParamViz(key) = pvObj;
      assert(~isempty(pvID));
      assert(obj.id2ParamViz(pvID)==pvObj);
    end
    
    function [tf,pvObj] = isprop(obj,prop)
      key = obj.fcnPropToKey(prop);
      m = obj.prop2ParamViz;
      tf = m.isKey(key);
      if tf
        pvObj = m(key);
      else
        pvObj = [];
      end
    end
    
    function [tf,pvObj] = ispropID(obj,pvID)
      m = obj.id2ParamViz;
      tf = m.isKey(pvID);
      if tf
        pvObj = m(pvID);
      else
        pvObj = [];
      end
    end

    function isOk = isValidParams(obj)
      
      isOk = getappdata(obj.hFig,'isOk');
      isOk = isempty(isOk) || isOk;      
      
    end
    
    function propSelected(obj,prop,ts)
      % Called when a prop is initially selected
      % 
      % prop: grid java prop
            
      [tfIsProp,pvObjNew] = obj.isprop(prop);
      if ~obj.isValidParams(),
        ParameterVisualization.grayOutAxes(obj.hAx,'Invalid parameters selected.');
        return;
      end
      
      pvObjCurrSel = obj.paramVizSelected;
      if ~isempty(pvObjCurrSel)
        if tfIsProp && pvObjNew==pvObjCurrSel
          % We already have pvObj selected; not sure this branch can ever
          % occur
          return;
        else
          % Either the new prop isn't obj.isprop, or it's a different/new
          % prop
          pvObjCurrSel.propUnselected();
          obj.paramVizSelected = [];
        end
      end

      obj.cbkToggleParamViz(tfIsProp,true);
      if tfIsProp
        
        if ts < obj.cbkTS,
          global DEBUG_PROPERTIESGUI2;
          if ~isempty(DEBUG_PROPERTIESGUI2) && DEBUG_PROPERTIESGUI2 > 0,
            fprintf('PROPSELECTED CANCEL, %s < %s\n',datestr(ts,'HH:MM:SS.FFF'),datestr(obj.cbkTS,'HH:MM:SS.FFF'));
          end
          return;
        end
        
        obj.cbkTS = ts;
        
        lblObj = obj.lObj;
        sPrm = obj.getCurrentParamsInTree();
%         fprintf(1,'PVH calling propSelected\n');
        pvObjNew.propSelected(obj.hAx,lblObj,char(prop.getFullName()),sPrm);
        obj.paramVizSelected = pvObjNew;        
      else
        ParameterVisualization.grayOutAxes(obj.hAx,...
          'No visualization available.');
      end
    end
    
    function propUpdatedGeneral(obj,prop,ts)
      
      if ts < obj.cbkTS,
        global DEBUG_PROPERTIESGUI2;
        if ~isempty(DEBUG_PROPERTIESGUI2) && DEBUG_PROPERTIESGUI2 > 0,
          fprintf('PROPUPDATEDGENERAL CANCEL, %s < %s\n',datestr(ts,'HH:MM:SS.FFF'),datestr(obj.cbkTS,'HH:MM:SS.FFF'));
        end
        return;
      end
      
      obj.cbkTS = ts;
      
      if ~obj.isValidParams(),
        ParameterVisualization.grayOutAxes(obj.hAx,'Invalid parameters selected.');
        return;
      end
      [tf,pvObj] = obj.isprop(prop);
      if tf
%         fprintf(1,'PVH calling propUpdated\n');
        sPrm = obj.getCurrentParamsInTree();        
        pvObj.propUpdated(obj.hAx,obj.lObj,char(prop.getFullName()),sPrm);
      end
    end
    
    function propUpdatedSpinner(obj,prop,pvObj,spinnerEvt,propName,ts)
      % Called when prop's spinner is clicked. propertyTable.appData.mirror
      % has not been updated yet
      % 
      % pvObj: prop->pvObj
      
      if ts < obj.cbkTS,
        global DEBUG_PROPERTIESGUI2;
        if ~isempty(DEBUG_PROPERTIESGUI2) && DEBUG_PROPERTIESGUI2 > 0,
          fprintf('PROPUPDATEDSPINNER CANCEL, %s < %s\n',datestr(ts,'HH:MM:SS.FFF'),datestr(obj.cbkTS,'HH:MM:SS.FFF'));
        end
        return;
      end

      obj.cbkTS = ts;
      
      if ~obj.isValidParams(),
        %ParameterVisualization.grayOutAxes(obj.hAx,'Invalid parameters selected.');
        return;
      end

      %fprintf(1,'PVH calling propUpdatedSpinner\n');
      sPrm = obj.getCurrentParamsInTree(); % sPrm outdated relative to spinnerEvt.spinnerValue;
      val = spinnerEvt.spinnerValue;
      %get(prop,'UserData')
      pvObj.propUpdatedDynamic(obj.hAx,obj.lObj,propName,sPrm,val);
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