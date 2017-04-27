function islabeled = IsLabeled(labeledposcurr)

islabeled = reshape(any(any(~isnan(labeledposcurr),1),2),[1,size(labeledposcurr,3)]);
