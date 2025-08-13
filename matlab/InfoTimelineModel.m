classdef InfoTimelineModel < handle

  properties
    lObj  % Labeler object that created this model
  end
  
  properties (SetAccess=private)
    selectOn_  % scalar logical, if true, select "Pen" is down
    selectOnStartFrm_  % frame where selection started
  end
  
  properties (Dependent)
    selectOn % scalar logical, if true, select "Pen" is down
    selectOnStartFrm % frame where selection started
  end
  
  methods
    function obj = InfoTimelineModel(labeler)
      % labeler: Labeler object that owns this model
      obj.lObj = labeler;
      obj.selectOn_ = false;
      obj.selectOnStartFrm_ = [];
    end
    
    function v = get.selectOn(obj)
      v = obj.selectOn_;
    end
        
    function setSelectMode(obj, newValue, currFrame)
      obj.selectOn_ = newValue ;
      if newValue
        obj.selectOnStartFrm_ = currFrame ;
      else
        obj.selectOnStartFrm_ = [] ;
      end
    end

    function v = get.selectOnStartFrm(obj)
      v = obj.selectOnStartFrm_;
    end

    function selectInit(obj)
      obj.selectOn_ = false;
      obj.selectOnStartFrm_ = [];
    end  % function
  end  % methods  
end  % classdef
