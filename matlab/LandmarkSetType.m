classdef LandmarkSetType
  enumeration 
    % caution, order matters here for clients
    Label ('labelPointsPlotInfo')
    Prediction ('predPointsPlotInfo')
  end

  properties
    labelerPropPlotInfo
  end

  methods 
    function obj = LandmarkSetType(lprop)
      obj.labelerPropPlotInfo = lprop;
    end
    % function meth = updateColorLabelerMethod(obj)
    %   meth = sprintf('setLandmark%sColors',char(obj));
    % end
    % function meth = updateCosmeticsLabelerMethod(obj)
    %   meth = sprintf('setLandmark%sCosmetics',char(obj));
    % end
  end
end
