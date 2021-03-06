classdef DLNetType
  properties
    shortString
    prettyString
    paramFileShort
    
    mdlNamePat
    mdlGlobPat
    
    trkAuxFlds % [naux] struct array of auxiliary tracking fields with 
               % fields .trkfld and .label
    timelinePropList % [naux] struct array of tracker-specific properties 
                     % in format used by InfoTimeline
                     
    doesOccPred % in practice, if this is true, totally-occluded landmarks 
                % will be included in the stripped lbl as p=nan and tfocc=true.
                % (and this is the only effect)
  end
  properties (Dependent)
    nTrkAuxFlds
  end
  
  enumeration 
    mdn ('mdn','MDN',...
          struct('trkfld',{'pTrkconf' 'pTrkconf_unet' 'pTrkocc'}, ...
                 'label',{'conf_mdn' 'conf_unet' 'scr_occ'}),true,...
                 'deepnet-%d.index','deepnet-*.index')
    deeplabcut ('dlc','DeepLabCut', ...
                struct('trkfld',{'pTrkconf'},'label',{'confidence'}),true, ...
                'deepnet-%d.index','deepnet-*.index')
    dpk ('dpk','DeepPoseKit', ...
          struct('trkfld',cell(0,1),'label',[]),false, ...
          'deepnet-%08d.h5','deepnet-*.h5')
    unet ('unet','Unet', ...
          struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
          'deepnet-%d.index','deepnet-*.index')
    openpose ('openpose','OpenPose',...
              struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
              'deepnet-%d','deepnet-*')
    leap ('leap','LEAP', ...
          struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
                  'deepnet-%d','deepnet-*')
    %hg ('hg','HourGlass')
  end
  
  methods 
    function v = get.nTrkAuxFlds(obj)
      v = numel(obj.trkAuxFlds);
    end
  end
  methods 
    function obj = DLNetType(sstr,pstr,auxflds,tfoccpred,...
        mdlNamePat,mdlGlobPat)
      obj.shortString = sstr;
      obj.prettyString = pstr;
      obj.paramFileShort = sprintf('params_deeptrack_%s.yaml',sstr);
      obj.trkAuxFlds = auxflds;
      obj.timelinePropList = DLNetType.auxflds2PropList(auxflds);
      obj.doesOccPred = tfoccpred;
      obj.mdlNamePat = mdlNamePat;
      obj.mdlGlobPat = mdlGlobPat;
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
      % TODO Should just be regular method but right now net can be a char.
      % TODO Continue removing switchyard.
      
      if ischar(net)
        net = DLNetType.(net);
      end
      
      g1 = sprintf(net.mdlNamePat,iterCurr);
      switch net
        case DLNetType.openpose
          g = { ...
            g1
            'traindata*'
            };

        case DLNetType.leap
          g = { ...
            g1
            'initial_model.h5' %            'leap_train.h5'
            'traindata*'
            'training_info.mat'
            };
          
        case DLNetType.dpk
          g = { ...
            g1
            'deepnet.conf.pickle'
            'traindata*'
            };
          
        otherwise % case {DLNetType.unet DLNetType.mdn DLNetType.deeplabcut}
          g = { ...
            sprintf('deepnet-%d.*',iterCurr)  % ugh this is not an instance prop
            'deepnet_ckpt' % 'splitdata.json'
            'traindata*'
            };
      end
    end   
  end
end
    