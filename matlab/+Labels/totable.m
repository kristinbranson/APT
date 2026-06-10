function t = totable(s,imov)
  frm = s.frm;
  iTgt = s.tgt;
  p = s.p';
  pTS = s.ts';
  tfocc = s.occ';
  if exist('imov','var')==0
    t = table(frm,iTgt,p,pTS,tfocc);
  else
    mov = repmat(imov,numel(frm),1);
    t = table(mov,frm,iTgt,p,pTS,tfocc);
  end
end  % function
