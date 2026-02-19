function tflbled = labeledTgts(s,nf)
  % nf: maximum number of frames
  %
  % tflbled: [nf itgtmax] tflbled(f,itgt) is true if itgt is labeled at f
  if isempty(s.tgt)
    itgtmax = 0;
    tgts = ones(size(s.frm));
  else
    if max(s.tgt)==0
      itgtmax = 1;
      tgts = s.tgt+1;
    else
      itgtmax = max(s.tgt);
      tgts = s.tgt;
    end
  end      
  tflbled = false(nf,itgtmax);
  idx = sub2ind([nf itgtmax],s.frm,tgts);
  tflbled(idx) = true;
  %ntgt = sum(tflbled,2);
end  % function
