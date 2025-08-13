classdef InfoTimelineModel < handle

  properties
    lObj % Labeler object that created this model
  end
  
  properties (SetAccess=private)
    selectOn_ % scalar logical, if true, select "Pen" is down
    selectOnStartFrm % frame where selection started
  end
  
  properties (Dependent)
    selectOn % scalar logical, if true, select "Pen" is down
  end
  
  events
    didSetSelectOn % fired when selectOn changes
  end
  
  methods
    function obj = InfoTimelineModel(labeler)
      % labeler: Labeler object that owns this model
      obj.lObj = labeler;
      obj.selectOn_ = false;
      obj.selectOnStartFrm = [];
    end
    
    function v = get.selectOn(obj)
      v = obj.selectOn_;
    end
    
    function set.selectOn(obj, v)
      obj.selectOn_ = v;
      if v
        obj.selectOnStartFrm = obj.lObj.currFrame;
      else
        obj.selectOnStartFrm = [];
      end
      
      % Update UI through controller if it exists
      if ~isempty(obj.lObj.controller_) && ~isempty(obj.lObj.controller_.infoTLController)
        tlController = obj.lObj.controller_.infoTLController;
        if ~tlController.isinit
          if v
            tlController.hCurrFrame.LineWidth = 3;
            if tlController.isL
              tlController.hCurrFrameL.LineWidth = 3;
            end
          else
            tlController.hCurrFrame.LineWidth = 0.5;
            if tlController.isL
              tlController.hCurrFrameL.LineWidth = 0.5;
            end
            tlController.setLabelerSelectedFrames();
          end
        end
      end
      
      notify(obj, 'didSetSelectOn');
    end
  end
  
end