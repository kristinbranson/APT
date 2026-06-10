function [s,nfrmslbl,nfrmscompact] = compactall(s)
  frmsun = unique(s.frm);
  nfrmscompact = 0;
  for f=frmsun(:)'
    [s,tfchanged] = Labels.compact(s,f);
    nfrmscompact = nfrmscompact+tfchanged;
  end
  nfrmslbl = numel(frmsun);
end  % function
