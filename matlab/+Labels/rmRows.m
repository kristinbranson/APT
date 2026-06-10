function s = rmRows(s,predicateFcn,~)
  % predicateFcn: eg @isnan, @isinf
  % rmDispStr: eg 'partially-labeled', 'fully-occluded' resp
  
  tf = any(predicateFcn(s.p),1);
  nrm = nnz(tf);
  if nrm>0
%        warningNoTrace('Labeler:nanData','Not including %d %s rows.',nrm,rmDispStr);
    s.p(:,tf) = [];
    s.ts(:,tf) = [];
    s.occ(:,tf) = [];
    s.frm(tf,:) = [];
    s.tgt(tf,:) = [];
    assert(~isfield(s,'split'));
  end
end  % function
