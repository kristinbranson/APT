function [frms,tgts] = isPartiallyLabeledT(s,itgt,nold)
  if isnan(itgt) || isempty(itgt),
    istgt = true(size(s.tgt));
  else
    istgt = s.tgt == itgt;
  end
  islabeled = Labels.isLabelerPerPt(s);
  ispartial = istgt' & all(islabeled(1:nold,:),1) & ~any(islabeled(nold+1:end,:),1);
  frms = s.frm(ispartial);
  tgts = s.tgt(ispartial);
end  % function
