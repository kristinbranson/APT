function ntgt = getNumTgts(s,frm)
  tf = s.frm==frm;
  ntgt = nnz(tf);  
end  % function
