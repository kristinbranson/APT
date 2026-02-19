function n = lObjNLabeled(lObj,labelsfld,varargin)
  [movis,itgts,gt] = myparse(varargin,'movi',[],'itgt',[],'gt',[]);
  if isempty(movis),
    movis = 1:numel(lObj.(labelsfld));
    movis = reshape(movis,size(lObj.(labelsfld)));
  end
  if isempty(itgts),
    itgts = cell(size(movis));
  end
  if gt,
    movis = abs(movis);
  end
  n = cell(size(movis));
  for ii = 1:numel(movis),
    movi = movis(ii);
    if isempty(itgts{ii}),
      n{ii} = nnz(~all(isnan(lObj.(labelsfld){movi}.p),1));
    else
      n{ii} = zeros(size(itgts{ii}));
      for jj = 1:numel(itgts{ii}),
        itgt = itgts{ii}(jj);
        n{ii}(jj) = nnz(~all(isnan(lObj.(labelsfld){movi}.p(:,lObj.(labelsfld){movi}.tgt==itgt)),1));
      end
    end
  end
end  % function
