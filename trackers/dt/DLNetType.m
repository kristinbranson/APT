classdef DLNetType
  properties
    shortString
    prettyString
    paramFileShort
    
    trkAuxFlds % [naux] struct array of auxiliary tracking fields with 
               % fields .trkfld and .label
    timelinePropList % [naux] struct array of tracker-specific properties 
                     % in format used by InfoTimeline
  end
  properties (Dependent)
    nTrkAuxFlds
  end
  
  enumeration 
    mdn ('mdn','MDN',struct('trkfld',{'pTrkconf' 'pTrkconf_unet'}, ...
                            'label',{'conf_mdn' 'conf_unet'}))
    deeplabcut ('dlc','DeepLabCut', struct('trkfld',cell(0,1),'label',[]))
    unet ('unet','Unet',            struct('trkfld',cell(0,1),'label',[]))
    openpose ('openpose','OpenPose',struct('trkfld',cell(0,1),'label',[]))
    leap ('leap','LEAP',            struct('trkfld',cell(0,1),'label',[]))
    %hg ('hg','HourGlass')
  end
  
  methods 
    function v = get.nTrkAuxFlds(obj)
      v = numel(obj.trkAuxFlds);
    end
  end
  methods 
    function obj = DLNetType(sstr,pstr,auxflds)
      obj.shortString = sstr;
      obj.prettyString = pstr;
      obj.paramFileShort = sprintf('params_deeptrack_%s.yaml',sstr);
      obj.trkAuxFlds = auxflds;
      obj.timelinePropList = DLNetType.auxflds2PropList(auxflds);
    end
  end
  
  methods (Static)
    function s = auxflds2PropList(auxflds)
      % Create .timelinePropList; see comments in properties block
      %
      % s: struct array detailing traker-specific props
           
      % From landmark_features.yaml, NonNegative FeatureClass
      TRANSTYPES = {'none' 'mean' 'median' 'std' 'min' 'max'};
      COORDSYS = 'Global';
      
      s = EmptyLandmarkFeatureArray();
      for iaux=1:numel(auxflds)
        label = auxflds(iaux).label;
        for tt=TRANSTYPES,tt=tt{1}; %#ok<FXSET>
          s(end+1,1) = ConstructLandmarkFeature(label,tt,COORDSYS); %#ok<AGROW>
        end
      end
    end
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
    