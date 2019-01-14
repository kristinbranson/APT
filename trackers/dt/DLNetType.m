classdef DLNetType
  properties 
    prettyString
  end
  enumeration 
    unet ('Unet')
    mdn ('MDN')
    deeplabcut ('DeepLabCut')
    openpose ('OpenPose')
    leap ('LEAP')
  end
  methods 
    function obj = DLNetType(str)
      obj.prettyString = str;
    end
  end
end
    