function tci = fillInCustomStagesIfNeeded(rawTCI, varargin)
  % Used to fill in the types of the two stages in a custom two-stage tracker.
  % varargin{1} specifies stage 1, varargin{2} specifies stage 2.  Either can be
  % missing or [] if caller wishes to use defaults.
  % This is a pure function.

  % Deal with arguments
  if numel(varargin)>=1 ,
    stage1Spec = varargin{1} ;
  else
    stage1Spec = [] ;
  end
  if numel(varargin)>=2 ,
    stage2Spec = varargin{2} ;
  else
    stage2Spec = [] ;
  end

  if ~strcmp(rawTCI{1}, 'DeepTrackerTopDownCustom') ,
    % Typical case, not a custom top-down tracker
    tci = rawTCI ;
    return
  end
  
  % First element of result always same
  tci1 = 'DeepTrackerTopDownCustom' ;

  % Handle stage 1, the object-detection stage
  tci2 = apt.fillInCustomStage1(rawTCI{2}, stage1Spec) ;

  % Handle stage 2, the pose-estimation-stage
  tci3 = apt.fillInCustomStage2(rawTCI{3}, stage2Spec) ;
  
  % Result should be a valid (not dummy) TCI array
  tci4 = 'valid' ;
  tci5 = true ;

  % Package for return
  tci = {tci1, tci2, tci3, tci4, tci5} ;
end  % function
