classdef DLNetType
  properties 
    prettyString
    paramFileShort
  end
  enumeration 
    unet ('Unet','params_deeptrack_unet.yaml')
    mdn ('MDN','params_deeptrack_mdn.yaml')
    deeplabcut ('DeepLabCut','params_deeptrack_dlc.yaml')
    openpose ('OpenPose','params_deeptrack_openpose.yaml')
    leap ('LEAP','params_deeptrack_leap.yaml')
  end
  methods 
    function obj = DLNetType(str,pfile)
      obj.prettyString = str;
      obj.paramFileShort = pfile;
    end
  end
end
    