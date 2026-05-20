function [tf] = lObjGetIsLabeled(lObj,labelsfld,tbl,gt,fullylabeled)
  
  if nargin < 5,
    fullylabeled = false; % KB: set default to be any kps labeled
  end

  tf = false(height(tbl),1);
  for i = 1:numel(lObj.(labelsfld)),
    if gt,
      movi = -i;
    else
      movi = i;
    end
    idx = tbl.mov == movi;
    if ~any(idx),
      continue;
    end
    cc = Labels.CLS_MD();
    if lObj.maIsMA,
      frs = eval(sprintf('%s([tbl.frm(idx)])',cc));
      [ism,j] = ismember(frs,[lObj.(labelsfld){i}.frm],'rows');
    else
      frs = eval(sprintf('%s([tbl.frm(idx),tbl.iTgt(idx)])',cc));
      [ism,j] = ismember(frs,[lObj.(labelsfld){i}.frm,lObj.(labelsfld){i}.tgt],'rows');
    end
    idx = find(idx);
    idx = idx(ism);
    j = j(ism);
    if fullylabeled,
      % only count fully labeled
      tf(idx) = ~any(isnan(lObj.(labelsfld){i}.p(:,j))); 
    else
      % any kps labeled
      tf(idx) = ~all(isnan(lObj.(labelsfld){i}.p(:,j))); 
    end
  end      
end  % function
