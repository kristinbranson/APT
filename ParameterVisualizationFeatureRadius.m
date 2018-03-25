classdef ParameterVisualizationFeatureRadius < ParameterVisualizationFeature
  
%   properties
%     hPlot % vector of plot handles output from Features.visualize*. 
%           % Set/created during init
%   end
  
  methods
    
    function init(obj,hAx,lObj,propFullName,sPrm)
      obj.initBase(hAx,lObj,sPrm,propFullName);
    end

    function update(obj,hAx,lObj,propFullName,sPrm)
      obj.updateBase(hAx,lObj,sPrm,propFullName);
    end

    function updateNewVal(obj,hAx,lObj,propFullName,sPrm,rad)
      sPrm.ROOT.CPR.Feature.Radius = rad;
      obj.updateBase(hAx,lObj,sPrm,propFullName);
    end
    
  end
  
end