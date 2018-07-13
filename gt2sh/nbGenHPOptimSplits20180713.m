%% Lots of C+P from nbGenSplits20180416

%% Test
for eh={'easy' 'hard'},eh=eh{1}; %#ok<FXSET>
  outerfname = sprintf('hpo_outer3_%s.mat',eh);
  outer = loadSingleVariableMatfile(outerfname);
  szassert(outer,[4523 3]);
  for iOuter=1:3
    innerfname = sprintf('hpo_outer3_fold%02d_inner3_%s.mat',iOuter,eh);
    inner = loadSingleVariableMatfile(innerfname);
    
    tfoutertst = outer(:,iOuter);
    tfinnertst = false(4523,3);
    tfoutertrn = ~tfoutertst;
    ioutertrn = find(tfoutertrn);
    noutertrn = numel(ioutertrn);
    szassert(inner,[noutertrn 3]);
    for iInner=1:3
      tfinnertst(ioutertrn(inner(:,iInner)),iInner) = true;
    end
    
    splitsumm = [tfinnertst tfoutertst];
    
    fprintf(1,'%s, outer split %d\n',eh,iOuter);
    sum(splitsumm,1)
    unique(sum(splitsumm,2))
  end
end

%% OUTER SPLIT: HARD
%% INNER SPLIT: HARD
NFOLD = 3;
for iOuterTstSplit=1:3
  tfTst = xvHPOOuter3Hard(:,iOuterTstSplit);  
  t = td.tMain20180503(~tfTst,:);
  fprintf(1,'Outer tst split %d: nTrn=%d\n',iOuterTstSplit,height(t));
  
  %%
  [s,tBalCats,tPrtCats,prtCat2Split] = stratifiedGroupSplit(NFOLD,t.lblCat,t.flyID);
  %%
  gBalC = categorical(t.lblCat);
  gBalC = reordercats(gBalC,tBalCats.cats);
  gBalFoldCounts = nan(height(tBalCats),NFOLD);
  for i=1:NFOLD
    tf = s==i;
    lblCatSplit = gBalC(tf);
    gBalFoldCounts(:,i) = countcats(lblCatSplit);
    fprintf('fold %d, n=%d\n',i,nnz(tf));
  end
  gBalFoldCounts./sum(gBalFoldCounts)
  %%
  flyUn = unique(t.flyID);
  for i=1:numel(flyUn)
    tf = flyUn(i)==t.flyID;
    split = unique(s(tf));
    assert(isscalar(split));
    fprintf(1,'fly %d, split %d\n',flyUn(i),split);  
  end

  %%
  xvHPOInner3Hard = false(height(t),NFOLD);
  for ifold=1:NFOLD
    xvHPOInner3Hard(:,ifold) = s==ifold;
  end
  sum(xvHPOInner3Hard,1)
  unique(sum(xvHPOInner3Hard,2))
  
  fname = sprintf('hpo_outer3_fold%02d_inner3.mat',iOuterTstSplit);
  save(fname,'xvHPOInner3Hard');
end

%% OUTER: EASY
%% INNER: EASY
NFOLD = 3;
for iOuterTstSplit=1:3
  tfTst = xvHPOOuter3Easy(:,iOuterTstSplit);  
  t = td.tMain20180503(~tfTst,:);
  fprintf(1,'Outer tst split %d: nTrn=%d\n',iOuterTstSplit,height(t));

  % t = td.tMain20180503;
  
  id = strcat(t.lblCat,'#',numarr2trimcellstr(t.flyID),'#',t.movID);
  idC = categorical(id);
  tCats = sortedsummary(idC)
  idC = reordercats(idC,tCats.cats);
  
  c = cvpartition(idC,'kfold',NFOLD);
  parts = arrayfun(@(z)test(c,z),1:NFOLD,'uni',0);
  parts = cat(2,parts{:});
  szassert(parts,[height(t) NFOLD]); % col i is indicator vec for part i
  unique(sum(parts,2))
  sum(parts,1)
  
  xvHPOInner3Easy = parts;
  fname = sprintf('hpo_outer3_fold%02d_inner3_easy.mat',iOuterTstSplit);
  save(fname,'xvHPOInner3Easy');
end