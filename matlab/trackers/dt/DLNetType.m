classdef DLNetType < handle
  % Deep Learning Network
  %
  % A DLNetType represents a particular deep net. It has a particular
  % architecture, hyperparameters, calling syntax for train/track, 
  % model/artifact structure when training, and set of outputs after
  % inference.
  %
  % When any DL job is run, one particular DLNetType is in play.
  
  properties (Constant)
    NETS = lclReadNetYaml();
  end
  properties
    shortString
    %paramString % field used in tracking params
    displayString
    %prettyString
    %paramFileShort
    
    mdlNamePat 
    mdlGlobPat
    modelGlobs
    
    trkAuxFields % [naux]
    trkAuxLabels % [naux]          
    timelinePropList % [naux] struct array of tracker-specific properties 
                     % in format used by InfoTimeline
                     
    doesOccPred % in practice, if this is true, totally-occluded landmarks 
                % will be included in the stripped lbl as p=nan and tfocc=true.
                % (and this is the only effect)
    isMultiAnimal % TODO: rename to "isMA bottom up" or similar
    %docker % docker tag
    %sing % path to singularity image
  end
  
  enumeration 
    mdn_joint_fpn ('mdn_joint_fpn')
    mmpose ('mmpose')  % If we had a time machine we would change this to mspn, but it's hard to do now (Aug 2023)
    deeplabcut ('deeplabcut')
    dpk ('dpk')
    openpose ('openpose')
    mdn ('mdn')
    unet ('unet')
    hrnet ('hrnet')
    %leap ('leap')
    multi_mdn_joint_torch ('multi_mdn_joint_torch')
    multi_openpose ('multi_openpose')
    detect_mmdetect ('detect_mmdetect')
    hrformer ('hrformer')
    multi_cid ('multi_cid')
    multi_dekr ('multi_dekr')
  end
  
  methods 
    function obj = DLNetType(key)
      q = DLNetType.NETS;
      s = q.(key);
      fns = fieldnames(s);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        %addprop(obj,f);
        switch f
          case 'modelCheckpointPat'
            v = s.(f);
            obj.mdlNamePat = v;
            obj.mdlGlobPat = regexprep(v,'%[0-9]*d','*');
          otherwise            
            obj.(f) = s.(f);
        end
      end
            
      obj.timelinePropList = DLNetType.auxflds2PropList(obj.trkAuxLabels);
    end
    function g = getModelGlobs(obj,iterCurr)
      g = cellfun(@(x)sprintf(x,iterCurr),obj.modelGlobs,'uni',0);
    end
    function tf = requiresTrnPack(obj, netMode)  %#ok<INUSD> 
      % whether training requires trnpack generation      
      tf = true ;
      return      
%       tf = obj.isMultiAnimal || ...
%           (netMode~=DLNetMode.singleAnimal && ...
%            netMode~=DLNetMode.multiAnimalTDPoseTrx);      
    end
  end
  
  methods (Static)
    
    function s = auxflds2PropList(auxlbls)
      % Create .timelinePropList; see comments in properties block
      %
      % s: struct array detailing traker-specific props
           
      % From landmark_features.yaml, NonNegative FeatureClass
      TRANSTYPES = {'none' 'mean' 'median' 'std' 'min' 'max'};
      COORDSYS = 'Global';
      
      s = EmptyLandmarkFeatureArray();
      for iaux=1:numel(auxlbls)
        label = auxlbls{iaux};
        for tt=TRANSTYPES,tt=tt{1}; %#ok<FXSET>
          s(end+1,1) = ConstructLandmarkFeature(label,tt,COORDSYS); %#ok<AGROW>
        end
      end
    end
  end
end  % classdef 
  
function s = lclReadNetYaml()
  netsYamlFilePath = fullfile(APT.Root, 'matlab', 'trackers', 'dt', 'nets.yaml') ;
  s = yaml.ReadYaml(netsYamlFilePath);
end
