classdef ShapeAugOrientation
  enumeration
    RAW  % Shapes are used as-is
    RANDOMIZED % Shapes are randomized to an arbitrary orientation
    SPECIFIED % Shapes are oriented to known orientations theta
  end
  
  methods (Static)
    % Factory utility, create SAO based on CPR params
    function obj = createPerParams(rotCorrectionUse,useTrxOrientation,...
        orientationThetas)
      if rotCorrectionUse
        if useTrxOrientation
          obj = ShapeAugOrientation.SPECIFIED;
          assert(~isempty(orientationThetas));
        else
          obj = ShapeAugOrientation.RANDOMIZED;
        end
      else
        obj = ShapeAugOrientation.RAW;
      end
    end
  end
end