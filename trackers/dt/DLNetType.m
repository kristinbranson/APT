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
  
  methods (Static)
    function g = modelGlobs(net,iterCurr)
      % net specific model globs in modelChainDir
      %
      % Future design unclear
      
      if ischar(net)
        net = DLNetType.(net);
      end
      
      switch net
        case {DLNetType.unet DLNetType.mdn DLNetType.deeplabcut}
          
          g = { ...
            sprintf('deepnet-%d.*',iterCurr) % latest iter
            'deepnet_ckpt' % 'splitdata.json'
            'traindata*'
            };
         
        case DLNetType.openpose
          g = { ...
            sprintf('deepnet-%d',iterCurr) % latest iter
            'traindata*'
            };

        case DLNetType.leap
          g = { ...
            sprintf('deepnet-%d',iterCurr) % latest iter
            'initial_model.h5'
            'leap_train.h5'
            'traindata*'
            'training_info.mat'
            };
          
      end
    end    
  end
end
    