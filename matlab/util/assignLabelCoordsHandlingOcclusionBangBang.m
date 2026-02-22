function assignLabelCoordsHandlingOcclusionBangBang(hPts, hTxt, xy, txtOffset)
% Assign label coordinates to point and text graphics handles, treating
% inf-valued coordinates as fully occluded (rendered as NaN).
%
% hPts: [npts]
% hTxt: [npts]
% xy: [npts x 2]

[npts, d] = size(xy);
assert(d==2);
assert(isequal(npts, numel(hPts), numel(hTxt)));

% FullyOccluded
tfIsOccluded = any(isinf(xy), 2);
setPositionsOfLabelLinesAndTextsBangBang(...
  hPts(~tfIsOccluded), hTxt(~tfIsOccluded), xy(~tfIsOccluded,:), txtOffset);
setPositionsOfLabelLinesAndTextsBangBang(...
  hPts(tfIsOccluded), hTxt(tfIsOccluded), nan(nnz(tfIsOccluded), 2), txtOffset);

end % function
