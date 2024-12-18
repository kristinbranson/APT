function [isInputModeInfoOK, outputModeInfo] = fixPrevModeInfo(labeler, paMode, inputModeInfo)
  % From the current labeler state, and the arguments, try to determine a new
  % paModeInfo stucture that is valid (I think).  On return, isInputModeInfoOK
  % is true iff the inputModeInfo was acceptable as-is.  If isInputModeInfoOK is
  % true, then outputModeInfo will be identical to inputModeInfo.  Otherwise,
  % outputModeInfo will (in theory) be a fixed version of inputModeInfo.
  % labeler is not mutated at all.

  % Deal with args
  if nargin < 2,
    paMode = labeler.prevAxesMode;
    inputModeInfo = labeler.prevAxesModeInfo;
  end
  
  % If paMode is PrevAxesMode.LASTSEEN, nothing to do
  if paMode ~= PrevAxesMode.FROZEN ,
    isInputModeInfoOK = true;
    outputModeInfo = inputModeInfo ;    
    return
  end
    
  % make sure the previous frame is labeled
  isInputModeInfoOK = false;
  lpos = labeler.labels;
  if (numel(lpos)<1) && (labeler.gtIsGTMode) && numel(labeler.labelsGT)>0
    lpos = labeler.labelsGT;
  end
  if isPrevAxesModeInfoValid(inputModeInfo),
    if numel(lpos) >= inputModeInfo.iMov,
      if isfield(inputModeInfo,'iTgt'),
        iTgt = inputModeInfo.iTgt;
      else
        iTgt = 1;
      end
      isInputModeInfoOK = Labels.isLabeledFT(lpos{inputModeInfo.iMov},inputModeInfo.frm,iTgt);
    end
    if isInputModeInfoOK,
      outputModeInfo = inputModeInfo ;
      return
    end
  end
  
  outputModeInfo = inputModeInfo ;
  if ~isfield(outputModeInfo,'axes_curr'),
    outputModeInfo.axes_curr = labeler.determinePrevAxesProperties(outputModeInfo);
  end
  
  [tffound,iMov,frm,iTgt] = labelFindOneLabeledFrameEarliest(labeler);
  if ~tffound,
    outputModeInfo.frm = [];
    outputModeInfo.iTgt = [];
    outputModeInfo.iMov = [];
    outputModeInfo.gtmode = false;
    return
  end
  outputModeInfo.frm = frm;
  outputModeInfo.iTgt = iTgt;
  outputModeInfo.iMov = iMov;
  outputModeInfo.gtmode = labeler.gtIsGTMode;
  
  tempPaModeInfo = labelerSetPrevMovieInfo(labeler, outputModeInfo);
  outputModeInfo = getDefaultPrevAxes(labeler, tempPaModeInfo);
end  % function
