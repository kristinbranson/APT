function frms = isLabeledT(s,itgt)
  % Find labeled frames (if any) for target itgt
  %
  % Pass itgt==nan to mean "any target"
  %
  % frms: [nfrmslbled] vec of frames that are labeled for target itgt.
  %   Not guaranteed to be in any order
  
  if isnan(itgt) || isempty(itgt)
    frms = unique(s.frm);
  else
    tf = s.tgt==itgt;
    frms = s.frm(tf);
  end
end  % function
