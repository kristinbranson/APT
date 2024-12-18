function outputModeInfo = labelerSetPrevMovieInfo(labeler, inputModeInfo, viewi)
  % The returned outputModeInfo is like inputModeInfo, but it has the .im field
  % and associated fields populated properly based on the other values in
  % inputModeInfo.
  % In spite of the name, this function does not modify labeler.
  if ~labeler.hasMovie || ~isPrevAxesModeInfoValid(inputModeInfo),
    outputModeInfo = inputModeInfo ;
    return
  end
  if nargin<3
    viewi = 1;
  end
  [im,isrotated,xdata,ydata,A,tform] = labelerGetTargetIm(labeler, inputModeInfo.iMov, inputModeInfo.frm, inputModeInfo.iTgt, viewi);
  outputModeInfo = inputModeInfo ;
  outputModeInfo.im =  im;
  outputModeInfo.isrotated =  isrotated;
  outputModeInfo.xdata =  xdata;
  outputModeInfo.ydata =  ydata;
  outputModeInfo.A =  A;
  outputModeInfo.tform =  tform;
end

