function result = fillInCustomStage2(rawTCI3, stage2Spec)
  % Used to fill in the type of stage 2 in a custom two-stage tracker.
  % This is a pure function.

  if ~isempty(stage2Spec)
    if iscell(stage2Spec) ,
      stage2ConstructorArgs = stage2Spec ;
      result = stage2ConstructorArgs ;
    elseif isa(stage2Spec, 'DLNetType') ,
      netType = stage2Spec ;
      result = horzcat(rawTCI3, {'trnNetType', netType}) ;
    elseif isstringy(stage2Spec) 
      netType = DLNetType(stage2Spec) ;
      result = horzcat(rawTCI3, {'trnNetType', netType}) ;
    else
      error('Stage 2 specification is not of a legal type.  It is a %s', class(stage2Spec)) ;            
    end
  else
    % stage2Spec is empty
    stage2TrnNetMode = rawTCI3{2} ;
    if stage2TrnNetMode == DLNetMode.multiAnimalTDPoseObj
      % bbox
      netType = DLNetType.mdn_joint_fpn ;
      result = horzcat(rawTCI3, {'trnNetType', netType}) ;
    elseif stage2TrnNetMode == DLNetMode.multiAnimalTDPoseHT
      % head-tail
      netType = DLNetType.mdn_joint_fpn ; 
      result = horzcat(rawTCI3, {'trnNetType', netType}) ;
    else
      error('Stage 2 mode is not a legal value') ;
    end
  end
end  % function

