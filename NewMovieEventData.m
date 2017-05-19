classdef (ConstructOnLoad) NewMovieEventData < event.EventData
  properties
    isFirstMovieOfProject % scalar logical
  end  
  methods
    function obj = NewMovieEventData(tf)
      obj.isFirstMovieOfProject = tf;
    end
  end
end