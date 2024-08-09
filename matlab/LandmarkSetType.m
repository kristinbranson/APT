classdef LandmarkSetType
  enumeration 
    % caution, order matters here for clients
    Label ('labelPointsPlotInfo')
    Prediction ('predPointsPlotInfo')
    Imported ('impPointsPlotInfo')
  end
  properties
    labelerPropPlotInfo
  end
  methods 
    function obj = LandmarkSetType(lprop)
      obj.labelerPropPlotInfo = lprop;
    end
    function meth = updateColorLabelerMethod(obj)
      meth = sprintf('updateLandmark%sColors',char(obj));
    end
    function meth = updateCosmeticsLabelerMethod(obj)
      meth = sprintf('updateLandmark%sCosmetics',char(obj));
    end
  end
end