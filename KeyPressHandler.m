classdef KeyPressHandler < handle
  % UIs or objects that want to get a shot at keypresses need to:
  % i) Be added to LabelerGUI/depHandles
  % ii) have a KeyPressHandler in their appdata.keyPressHandler.
  
  methods
    
    function tfHandled = handleKeyPress(obj,evt)
      % evt: keypress event, eg
      %   matlab.ui.eventdata.UIClientComponentKeyEvent
      %
      % tfHandled: logical scalar, if true, keypress was used/consumed
      % here, otherwise, it was not used
      
    end
    
  end
  
end