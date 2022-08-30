function swapKeypoints(lObj,swaps)

labelsnew = lObj.labels;
npts = size(labelsnew{1}.p,1)/2;

for i = 1:numel(labelsnew),
  n = size(labelsnew{i}.p,2);
  p = reshape(labelsnew{i}.p,[npts,2,n]);
  for j = 1:size(swaps,1),
    swap = swaps(j,:);
    tmp = p(swap(1),:,:);
    p(swap(1),:,:) = p(swap(2),:,:);
    p(swap(2),:,:) = tmp;
    labelsnew{i}.p = reshape(p,[2*npts,n]);
    tmp = labelsnew{i}.ts(swap(1),:);
    labelsnew{i}.ts(swap(1),:) = labelsnew{i}.ts(swap(2),:);
    labelsnew{i}.ts(swap(2),:) = tmp;
    tmp = labelsnew{i}.occ(swap(1),:);
    labelsnew{i}.occ(swap(1),:) = labelsnew{i}.occ(swap(2),:);
    labelsnew{i}.occ(swap(2),:) = tmp;
  end
end
lObj.labels = labelsnew;