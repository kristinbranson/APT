classdef LandmarkSetType
  enumeration 
    % caution, order matters here for clients
    Label
    Prediction 
    Imported
  end
  methods 
    function meth = updateColorLabelerMethod(obj)
      meth = sprintf('updateLandmark%sColors',char(obj));
    end
    function meth = updateCosmeticsLabelerMethod(obj)
      meth = sprintf('updateLandmark%sCosmetics',char(obj));
    end
  end
end