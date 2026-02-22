function setPositionsOfLabelLinesAndTextsToNanBangBang(hPts, hTxt)
% Set pts/txt to be "offscreen" ie positions to NaN.
TXTOFFSET_IRRELEVANT = 1;
setPositionsOfLabelLinesAndTextsBangBang(hPts, hTxt, ...
  nan(numel(hPts),2), TXTOFFSET_IRRELEVANT);
end % function
