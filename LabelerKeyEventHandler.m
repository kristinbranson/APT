classdef LabelerKeyEventHandler < handle
  properties
    evtMatchFcn % fcn with sig: tf=evtMatchFcn(evt) where evt is a keypress 
                % event. Return true to handle given keypress
    actionFcn % fcn with sig: actionFcn(lObj,evt). lObj is Labeler obj, evt
              % is event that matched. Take desired action
  end
  
  methods
    function obj = LabelerKeyEventHandler(eFcn,actFcn)
      obj.evtMatchFcn = eFcn;
      obj.actionFcn = actFcn;
    end
    function tfHandled = handleKeyPress(obj,evt,lObj)
      % evt: keypress event, eg
      %   matlab.ui.eventdata.UIClientComponentKeyEvent
      
      tfHandled = obj.evtMatchFcn(evt);
      if tfHandled
        obj.actionFcn(lObj,evt);
      end
    end
  end
end