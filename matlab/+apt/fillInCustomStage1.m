function result = fillInCustomStage1(rawTCI2, stage1Spec)
  % Used to fill in the type of stage 1 in a custom two-stage tracker.
  % This is a pure function.
  
  if ~isempty(stage1Spec) 
    if iscell(stage1Spec) ,
      stage1ConstructorArgs = stage1Spec ;
      result = stage1ConstructorArgs ;
    elseif isa(stage1Spec, 'DLNetType') ,
      netType = stage1Spec ;
      result = horzcat(rawTCI2, {'trnNetType', netType}) ;
    elseif isstringy(stage1Spec) 
      netType = DLNetType(stage1Spec) ;
      result = horzcat(rawTCI2, {'trnNetType', netType}) ;
    else
      error('Stage 1 specification is not of a legal type.  It is a %s', class(stage1Spec)) ;            
    end
  else
    % stage1Spec is empty    
    stage1TrnNetMode = rawTCI2{2} ;
    if stage1TrnNetMode == DLNetMode.multiAnimalTDDetectObj
      % bbox
      netType = DLNetType.detect_mmdetect ;
      result = horzcat(rawTCI2, {'trnNetType', netType}) ;
    elseif stage1TrnNetMode == DLNetMode.multiAnimalTDDetectHT
      % head-tail
      netType = DLNetType.multi_mdn_joint_torch ; 
      result = horzcat(rawTCI2, {'trnNetType', netType}) ;
    else
      error('Stage 1 mode is not a legal value') ;
    end
  end  
end
