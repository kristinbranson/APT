function tf = labeledFrames(s,nfrm)
  tf = false(nfrm,1);
  tf(s.frm) = true;
  assert(numel(tf)==nfrm);
end  % function
