classdef DLNetType < handle %dynamicprops
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
                     
    doesOccPred
    isMultiAnimal   
  end
  
  enumeration 
    mdn ('mdn')
    unet ('unet')
    mdn_joint_fpn ('mdn_joint_fpn')
    mmpose ('mmpose')
    deeplabcut ('deeplabcut')
    dpk ('dpk')
    openpose ('openpose')
    leap ('leap')
    multi_mdn_joint_torch ('multi_mdn_joint_torch')
    multi_openpose ('multi_openpose')
  end
  
  methods 
    function obj = DLNetType(key)
      q = DLNetType.NETS;
      s = q.(key);
      fns = fieldnames(s);
      for f=fns(:)',f=f{1};
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
end 
  
function s = lclReadNetYaml()
yaml = fullfile(APT.Root,'matlab','trackers','dt','nets.yaml');
s = ReadYaml(yaml);
end