classdef DLNetType
  properties
    shortString
    prettyString
    paramFileShort
  end
  enumeration 
    mdn ('mdn','MDN')
    deeplabcut ('dlc','DeepLabCut')
    unet ('unet','Unet')
    openpose ('openpose','OpenPose')
    leap ('leap','LEAP')
    %hg ('hg','HourGlass')
  end
  methods 
    function obj = DLNetType(sstr,pstr)
      obj.shortString = sstr;
      obj.prettyString = pstr;
      obj.paramFileShort = sprintf('params_deeptrack_%s.yaml',sstr);
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
        case DLNetType.openpose
          g = { ...
            sprintf('deepnet-%d',iterCurr) % latest iter
            'traindata*'
            };

        case DLNetType.leap
          g = { ...
            sprintf('deepnet-%d',iterCurr) % latest iter
            'initial_model.h5' %            'leap_train.h5'
            'traindata*'
            'training_info.mat'
            };
          
        otherwise % case {DLNetType.unet DLNetType.mdn DLNetType.deeplabcut}
          g = { ...
            sprintf('deepnet-%d.*',iterCurr) % latest iter
            'deepnet_ckpt' % 'splitdata.json'
            'traindata*'
            };
      end
    end    
  end
end
    