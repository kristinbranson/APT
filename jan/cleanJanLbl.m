function cleanJanLbl(lblfile,lblfileNew)
l = load(lblfile,'-mat');
assert(exist(lblfileNew,'file')==0);
assert(numel(l.movieFilesAll)==1);
lpos = l.labeledpos{1};
lpostag = l.labeledpostag{1};
[npt,~,nfrm] = size(lpos);

% overview
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
tfLbled = nptsLbled>0;
fLbled = find(tfLbled);
nptsLblUn = unique(nptsLbled);
nptsLblUnCnt = arrayfun(@(x)nnz(nptsLbled==x),nptsLblUn);
fprintf(1,'npt=%d, nfrm=%d\n',npt,nfrm);
fprintf(1,'[nptsLblUn nptsLblUnCnt]:\n');
disp([nptsLblUn nptsLblUnCnt]);

% copy pts 123 from first pt to others
if 0
  f0 = find(nptsLbled==npt);
  assert(isscalar(f0),'Expected exactly one pt with all %d labeled pts.',npt);
  fset = setdiff(fLbled,f0);
  for f=fset(:)'
    tmp = lpos(1:3,:,f);
    assert(all(isnan(tmp(:))));
    lpos(1:3,:,f) = lpos(1:3,:,f0);
  end
  fprintf(1,'Copied pts 1,2,3 from frame f0=%d to %d other frames.\n',f0,numel(fset));
end

% clear partially-labeled frames
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
%tf0 = nPtsLbled==0;
%tfGood = nPtsLbled==npt;
tfRest = ismember(nptsLbled,1:npt-1);
nRest = nnz(tfRest);
lpos(:,:,tfRest) = nan;
lpostag(:,tfRest) = {[]};
fprintf(1,'cleared %d partially-labeled frames:\n',nRest);
disp(find(tfRest));
  
% overview
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
tfLbled = nptsLbled>0;
fLbled = find(tfLbled);
nptsLblUn = unique(nptsLbled);
nptsLblUnCnt = arrayfun(@(x)nnz(nptsLbled==x),nptsLblUn);
fprintf(1,'npt=%d. [nptsLblUn nptsLblUnCnt]:\n',npt);
disp([nptsLblUn nptsLblUnCnt]);

l.labeledpos{1} = lpos;
l.labeledpostag{1} = lpostag;
fprintf(1,'Saving to %s\n',lblfileNew);
save(lblfileNew,'-mat','-struct','l');