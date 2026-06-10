function [s,tfchanged,ntgts] = compact(s,frm)
  % Arbitrarily renames/remaps target indices for given frm to fall
  % into 1:ntgts. No consideration is given for continuity or
  % identification across frames.
  %
  % tfchanged: true if any change/edit was made to s
  % ntgts: number of tgts for frm
  
  tf = s.frm==frm;
  itgts = s.tgt(tf);
  ntgts = numel(itgts);
  tfchanged = ~isequal(sort(itgts),(1:ntgts)'); 
  % order currently never matters in s.frm, s.tgt
  if tfchanged
    s.tgt(tf) = (1:ntgts)';
  end
end  % function
