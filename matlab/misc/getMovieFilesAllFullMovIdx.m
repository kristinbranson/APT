function v = getMovieFilesAllFullMovIdx(labeler,mIdx)
  % mIdx: MovieIndex vector
  % v: [numel(mIdx)xnview] movieFilesAllFull/GT 
  
  assert(isa(mIdx,'MovieIndex'));
  [iMov,gt] = mIdx.get();
  n = numel(iMov);
  v = cell(n,labeler.nview);
  mfaf = labeler.movieFilesAllFull;
  mfafGT = labeler.movieFilesAllGTFull;
  for i=1:n
    if gt(i)
      v(i,:) = mfafGT(iMov(i),:);
    else
      v(i,:) = mfaf(iMov(i),:);
    end
  end
end
