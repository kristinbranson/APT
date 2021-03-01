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
                     
    doesOccPred
    doesMA   
  end
  properties (Dependent)
    nTrkAuxFlds
  end
  
  enumeration 
    mdn ('mdn','MDN',...
          struct('trkfld',{'pTrkconf' 'pTrkconf_unet' 'pTrkocc'}, ...
                 'label',{'conf_mdn' 'conf_unet' 'scr_occ'}),true,...
                 'deepnet-%d.index','deepnet-*.index',false)
    deeplabcut ('dlc','DeepLabCut', ...
                struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false, ...
                'deepnet-%d.index','deepnet-*.index',false)
    dpk ('dpk','DeepPoseKit', ...
          struct('trkfld',cell(0,1),'label',[]),false, ...
          'deepnet-%08d.h5','deepnet-*.h5',false)
    unet ('unet','Unet', ...
          struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
          'deepnet-%d.index','deepnet-*.index',false)
    openpose ('openpose','OpenPose',...
              struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
              'deepnet-%d','deepnet-*',false)
    leap ('leap','LEAP', ...
          struct('trkfld',{'pTrkconf'},'label',{'confidence'}),false,...
                  'deepnet-%d','deepnet-*',false)
    multi_mdn_joint_torch ('ma','MultiAnimal',...
                 struct('trkfld',cell(0,1),'label',[]),false,...
                        'deepnet-%d','deepnet-*',true)
  end
  
  methods 
    function v = get.nTrkAuxFlds(obj)
      v = numel(obj.trkAuxFlds);
    end
  end
  methods 
    function obj = DLNetType(sstr,pstr,auxflds,tfoccpred,...
        mdlNamePat,mdlGlobPat,doesMA)
      obj.shortString = sstr;
      obj.prettyString = pstr;
      obj.paramFileShort = sprintf('params_deeptrack_%s.yaml',sstr);
      obj.trkAuxFlds = auxflds;
      obj.timelinePropList = DLNetType.auxflds2PropList(auxflds);
      obj.doesOccPred = tfoccpred;
      obj.mdlNamePat = mdlNamePat;
      obj.mdlGlobPat = mdlGlobPat;
      obj.doesMA = doesMA;
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
        case DLNetType.multi_mdn_joint_torch
          g = { ...
            g1 
            'traindata*'
            'deepnet_ckpt'
            '*json'
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
    