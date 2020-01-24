function idxselect = SelectRandomButSpacedFrames(canselect,nselect,mindist)

idxselect = nan(1,nselect);
for i = 1:nselect,
  
  idxselect(i) = randsample(find(canselect),1);
  canselect(max(1,idxselect(i)-mindist+1):min(numel(canselect),idxselect(i)+mindist-1)) = false;
  assert(any(canselect));
    
end