classdef LabelTracker < handle
  % LabelTracker knows how to take a bunch of images+labels and learn a
  % classifier to predict/track labels on new images.
  
  properties    
    lObj % (back)handle to Labeler object    
  end
  
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;
    end
    
    function track(obj)
    end
    
  end
  
end