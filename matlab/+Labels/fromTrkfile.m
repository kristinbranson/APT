function s = fromTrkfile(trk)
  if isfield(trk,'pTrkiPt')
    assert(isequal(trk.pTrkiPt(:)',1:size(trk.pTrk,1)),...
      'Unexpected point specification in .pTrkiPt.');
  end
  if trk.isfull
    s = Labels.fromarray(trk.pTrk,'lposTS',trk.pTrkTS,...
      'lpostag',trk.pTrkTag,'frms',trk.pTrkFrm,'tgts',trk.pTrkiTgt);
  else
    s = Labels.fromtable(trk.tableform('labelsColNames',true));
  end
end  % function
