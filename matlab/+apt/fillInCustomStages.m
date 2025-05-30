function tci = fillInCustomStages(rawTCI, varargin)
  % Used to fill in the types of the two stages in a custom two-stage tracker.
  % varargin{1} specifies stage 1, varargin{2} specifies stage 2.  Either can be
  % missing or [] if caller wishes to use defaults.
  % This is a pure function.

  tci = rawTCI ;

  % Handle stage 1, the object-detection stage
  if numel(varargin)>=1 ,
    stage1Spec = varargin{1} ;
  else
    stage1Spec = [] ;
  end
  if ~isempty(stage1Spec) 
    if iscell(stage1Spec) ,
      stage1ConstructorArgs = stage1Spec ;
      tci{2} = stage1ConstructorArgs ;
    elseif isa(stage1Spec, 'DLNetType') ,
      netType = stage1Spec ;
      tci{2} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    elseif isstringy(stage1Spec) 
      netType = DLNetType(stage1Spec) ;
      tci{2} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    else
      error('Stage 1 specification is not of a legal type.  It is a %s', class(stage1Spec)) ;            
    end
  else
    % stage1Spec is empty    
    stage1TrnNetMode = rawTCI{2}{2} ;
    if stage1TrnNetMode == DLNetMode.multiAnimalTDDetectObj
      % bbox
      netType = DLNetType.detect_mmdetect ;
      tci{2} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    elseif stage1TrnNetMode == DLNetMode.multiAnimalTDDetectHT
      % head-tail
      netType = DLNetType.multi_mdn_joint_torch ; 
      tci{2} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    else
      error('Stage 1 mode is not a legal value') ;
    end
  end

  % Handle stage 2, the pose-esitmation-stage
  if numel(varargin)>=2 ,
    stage2Spec = varargin{2} ;
  else
    stage2Spec = [] ;
  end
  
  if ~isempty(stage2Spec)
    stage2Spec = varargin{2} ;
    if iscell(stage2Spec) ,
      stage2ConstructorArgs = stage2Spec ;
      tci{3} = stage2ConstructorArgs ;
    elseif isa(stage2Spec, 'DLNetType') ,
      netType = stage2Spec ;
      tci{3} = horzcat(rawTCI{3}, {'trnNetType', netType}) ;
    elseif isstringy(stage2Spec) 
      netType = DLNetType(stage2Spec) ;
      tci{3} = horzcat(rawTCI{3}, {'trnNetType', netType}) ;
    else
      error('Stage 2 specification is not of a legal type.  It is a %s', class(stage2Spec)) ;            
    end
  else
    % stage2Spec is empty
    stage2TrnNetMode = rawTCI{3}{2} ;
    if stage2TrnNetMode == DLNetMode.multiAnimalTDPoseObj
      % bbox
      netType = DLNetType.mdn_joint_fpn ;
      tci{3} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    elseif stage2TrnNetMode == DLNetMode.multiAnimalTDPoseHT
      % head-tail
      netType = DLNetType.mdn_joint_fpn ; 
      tci{3} = horzcat(rawTCI{2}, {'trnNetType', netType}) ;
    else
      error('Stage 2 mode is not a legal value') ;
    end
  end
  
  % Finally, set to valid
  tci{4} = 'valid' ;
  tci{5} = true ;
end  % function
