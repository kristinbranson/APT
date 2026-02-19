function itgts = isLabeledF(s,frm)
  % Find labeled targets (if any) for frame frm
  %
  % itgts: [ntgtlbled] vec of targets that are labeled in frm
  
  tf = s.frm==frm;
  itgts = s.tgt(tf);
end  % function
