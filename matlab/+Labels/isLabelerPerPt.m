function [tf] = isLabelerPerPt(s)
  % [tf] = isLabelerPerPt(s)
  % Added by KB 20220206
  % tf(i,j) indicates whether landmark i is labeled for label j
  tf = permute(any(~isnan(reshape(s.p,[size(s.p,1)/2, 2, size(s.p,2)])),2),[1,3,2]);      
end  % function
