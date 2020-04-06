%% Lots of C+P from nbGenSplits20180416

%%%%%%%%%
%% SH %%%
%%%%%%%%%

%% Test
tMain = loadSingleVariableMatfile('tMain4523.mat');
for eh={'easy' 'hard'},eh=eh{1}; %#ok<FXSET>
  outerfname = sprintf('hpo_outer3_%s.mat',eh);
  outer = loadSingleVariableMatfile(outerfname);
  szassert(outer,[4523 3]);
  for iOuter=1:3
    innerfname = sprintf('hpo_outer3_%s_fold%02d_inner3.mat',eh,iOuter);
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
    
    tMainOuterTrn = tMain(tfoutertrn,:);
    tMainOuterTst = tMain(tfoutertst,:);
    outertrnTableName = sprintf('hpo_outer3_%s_fold%02d_tblTrn.mat',eh,iOuter);
    outertstTableName = sprintf('hpo_outer3_%s_fold%02d_tblTst.mat',eh,iOuter);
%     save(outertrnTableName,'tMainOuterTrn');
%     fprintf(1,'Saved outer train table (nTrn=%d): %s\n',height(tMainOuterTrn),outertrnTableName);
    save(outertstTableName,'tMainOuterTst');
    fprintf(1,'Saved outer test table (nTst=%d): %s\n',height(tMainOuterTst),outertstTableName);
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


%%%%%%%%%
%% AR %%%
%%%%%%%%%
%% OUTER: EASY
%% INNER: EASY
tblTrnAll = loadSingleVariableMatfile('tblTrn4703.mat');
tblTrnAll = tblTrnAll(:,MFTable.FLDSID);
flyidsC = categorical(tblTrnAll.mov).*categorical(tblTrnAll.iTgt);
flyidsC = removecats(flyidsC);
tcats = sortedsummary(flyidsC)
flyidsC = reordercats(flyidsC,tcats.cats);

n = height(tblTrnAll);
BIGKFOLD = 9;
c = cvpartition(flyidsC,'kfold',BIGKFOLD);
bigparts = arrayfun(@(z)test(c,z),1:BIGKFOLD,'uni',0);
bigparts = cat(2,bigparts{:});
szassert(bigparts,[n BIGKFOLD]); % col i is indicator vec for part i
unique(sum(bigparts,2))
sum(bigparts,1)
for k=1:BIGKFOLD
  tfk = bigparts(:,k);
  fprintf(1,'Fold %d\n',k);
  sortedsummary(flyidsC(tfk))
end
%%
for iOuterTstSplit=1:3
  kOuterTst = (1:3) + (iOuterTstSplit-1)*3;
  kOuterTrn = setdiff(1:9,kOuterTst);
  tfOuterTst = any(bigparts(:,kOuterTst),2);
  bigPartsOuterTrn = bigparts(:,kOuterTrn);
  tfOuterTrn = any(bigPartsOuterTrn,2);
  assert(all(tfOuterTst+tfOuterTrn==1));
  
  tblOuterTst = tblTrnAll(tfOuterTst,:);
  tblOuterTrn = tblTrnAll(tfOuterTrn,:);
  innerparts = bigPartsOuterTrn(tfOuterTrn,:);
  assert(size(innerparts,2)==6);
  
  innerSplit = innerparts(:,[1 3 5]) + innerparts(:,[2 4 6]);
  innerSplit = logical(innerSplit);
  
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  assert(isempty(intersect(tblOuterTst,tblOuterTrn)));
  szassert(innerSplit,[nOuterTrn 3]);
  assert(all(sum(innerSplit,2)==1));
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));  
  flyidsOuterTrnC = flyidsC(tfOuterTrn);
  for kinner=1:3
    fprintf(1,'Inner split %d:\n',kinner);
    sortedsummary(flyidsOuterTrnC(innerSplit(:,kinner)))
  end
  
  fnameTblOuterTst = sprintf('arhpo_outer3_easy_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('arhpo_outer3_easy_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('arhpo_outer3_easy_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end

%% OUTER: HARD
%% INNER: HARD
% For "hard" we do a strict partitioning by movie, ie no movie appears in
% more than one inner or outer split. This leads to a "rougher" partioning
% that deviates from the 9-way bigpartition but that's ok. The last split
% in particular is really imbalanced in the inner split.
folddefs = struct(...
  'outerTstIMovs',{[4 5],1,[2 3]},...
  'innerTrnIMovs',{ {1 2 3}, {2 4 [3 5]}, {1 4 5} });
for iOuterTstSplit=1:3
  sdef = folddefs(iOuterTstSplit);
  tfOuterTst = ismember(tblTrnAll.mov,sdef.outerTstIMovs);
  tfOuterTrn = ~tfOuterTst;
  tfInner = cellfun(@(imovs)ismember(tblTrnAll.mov,imovs),sdef.innerTrnIMovs,'uni',0);
  assert(numel(tfInner)==3);
  tfInner = cat(2,tfInner{:});
  
  assert(isequal(any(tfInner,2),tfOuterTrn));
  
  assert(all(tfOuterTst+tfOuterTrn==1));  
  tblOuterTst = tblTrnAll(tfOuterTst,:);
  tblOuterTrn = tblTrnAll(tfOuterTrn,:);
  innerSplit = tfInner(tfOuterTrn,:);
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));
  fprintf('Outer Tst Movs: %s. Outer Trn movs: %s\n',...
    mat2str(unique(tblOuterTst.mov)),...
    mat2str(unique(tblOuterTrn.mov)));
  for kinner=1:3
    fprintf(1,'  Inner split %d movs: %s\n',kinner,...
      mat2str(unique(tblOuterTrn.mov(innerSplit(:,kinner)))));
  end

  fnameTblOuterTst = sprintf('arhpo_outer3_hard_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('arhpo_outer3_hard_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('arhpo_outer3_hard_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end

%%%%%%%%%%%%%%%
%% CAROLINE %%%
%%%%%%%%%%%%%%%
%% OUTER: EASY
%% INNER: EASY
tblTrnAll = loadSingleVariableMatfile('tblTrn295.mat');
tblTrnAll = tblTrnAll(:,MFTable.FLDSID);
movC = categorical(tblTrnAll.mov);
summary(movC)

n = height(tblTrnAll);
BIGKFOLD = 9;
c = cvpartition(movC,'kfold',BIGKFOLD);
bigparts = arrayfun(@(z)test(c,z),1:BIGKFOLD,'uni',0);
bigparts = cat(2,bigparts{:});
szassert(bigparts,[n BIGKFOLD]); % col i is indicator vec for part i
unique(sum(bigparts,2))
sum(bigparts,1)
for k=1:BIGKFOLD
  tfk = bigparts(:,k);
  fprintf(1,'Fold %d\n',k);
  sortedsummary(movC(tfk))
end
%%
for iOuterTstSplit=1:3
  kOuterTst = (1:3) + (iOuterTstSplit-1)*3;
  kOuterTrn = setdiff(1:9,kOuterTst);
  tfOuterTst = any(bigparts(:,kOuterTst),2);
  bigPartsOuterTrn = bigparts(:,kOuterTrn);
  tfOuterTrn = any(bigPartsOuterTrn,2);
  assert(all(tfOuterTst+tfOuterTrn==1));
  
  tblOuterTst = tblTrnAll(tfOuterTst,:);
  tblOuterTrn = tblTrnAll(tfOuterTrn,:);
  innerparts = bigPartsOuterTrn(tfOuterTrn,:);
  assert(size(innerparts,2)==6);
  
  innerSplit = innerparts(:,[1 3 5]) + innerparts(:,[2 4 6]);
  innerSplit = logical(innerSplit);
  
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  assert(isempty(intersect(tblOuterTst,tblOuterTrn)));
  szassert(innerSplit,[nOuterTrn 3]);
  assert(all(sum(innerSplit,2)==1));
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));  
  movsOuterTrnC = movC(tfOuterTrn);
  for kinner=1:3
    fprintf(1,'Inner split %d:\n',kinner);
    summary(movsOuterTrnC(innerSplit(:,kinner)))
  end
  
  fnameTblOuterTst = sprintf('larvahpo_outer3_easy_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('larvahpo_outer3_easy_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('larvahpo_outer3_easy_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end

%% OUTER: HARD
%% INNER: HARD
% For "hard" we hold one movie out, then the other 3 are the inner split.
movI = false(n,4);
for iOuterTstSplit=1:4
  movI(:,iOuterTstSplit) = tblTrnAll.mov==iOuterTstSplit;
end
for iOuterTstSplit=1:4
  tfOuterTst = movI(:,iOuterTstSplit);
  tfOuterTrn = ~tfOuterTst;
  tfInner = movI;
  tfInner(:,iOuterTstSplit) = [];  
  assert(isequal(any(tfInner,2),tfOuterTrn));  
  
  assert(all(tfOuterTst+tfOuterTrn==1));
  tblOuterTst = tblTrnAll(tfOuterTst,:);
  tblOuterTrn = tblTrnAll(tfOuterTrn,:);
  innerSplit = tfInner(tfOuterTrn,:);
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));
  fprintf('Outer Tst Movs: %s. Outer Trn movs: %s\n',...
    mat2str(unique(tblOuterTst.mov)),...
    mat2str(unique(tblOuterTrn.mov)));
  for kinner=1:3
    fprintf(1,'  Inner split %d movs: %s\n',kinner,...
      mat2str(unique(tblOuterTrn.mov(innerSplit(:,kinner)))));
  end

  fnameTblOuterTst = sprintf('larvahpo_outer4_hard_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('larvahpo_outer4_hard_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('larvahpo_outer4_hard_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end

%%
%%%%%%%%%%%%%%
%% BRITMAT %%%
%%%%%%%%%%%%%%

ttrn = loadSingleVariableMatfile('ttrn15343.mat');

% %% this was for the big outer xv 3-fold splits
% cvpmovmse = cvpartition(gmovmse,'kfold',3,'stratify',true);
% cvpmse = cvpartition(gmse,'kfold',3,'stratify',true);
% tftestmovmse = arrayfun(@(x)test(cvpmovmse,x),1:3,'uni',0);
% tftestmse = arrayfun(@(x)test(cvpmse,x),1:3,'uni',0);
% tftestmovmse = cat(2,tftestmovmse{:});
% tftestmse = cat(2,tftestmse{:});
% %%
% smse = stratifiedGroupSplit(3,ones(size(gmse)),gmse);
% tftestmsesplit = arrayfun(@(x)smse==x,1:3,'uni',0);
% tftestmsesplit = cat(2,tftestmsesplit{:});

%% OUTER: EASY
%% INNER: EASY

gmse = categorical(ttrn.mouse);
% don't use gmovmse in HPO, too easy
% gmovmse = categorical(ttrn.mov).*categorical(ttrn.mouse);
% gmovmse = removecats(gmovmse);

tcats = sortedsummary(gmse)
gmse = reordercats(gmse,tcats.cats);

n = height(ttrn);
BIGKFOLD = 9;
c = cvpartition(gmse,'kfold',BIGKFOLD,'stratify',true);
bigparts = arrayfun(@(z)test(c,z),1:BIGKFOLD,'uni',0);
bigparts = cat(2,bigparts{:});
szassert(bigparts,[n BIGKFOLD]); % col i is indicator vec for part i
unique(sum(bigparts,2))
sum(bigparts,1)
for k=1:BIGKFOLD
  tfk = bigparts(:,k);
  fprintf(1,'Fold %d\n',k);
  sortedsummary(gmse(tfk))
end
%%
for iOuterTstSplit=1:3
  kOuterTst = (1:3) + (iOuterTstSplit-1)*3;
  kOuterTrn = setdiff(1:9,kOuterTst);
  tfOuterTst = any(bigparts(:,kOuterTst),2);
  bigPartsOuterTrn = bigparts(:,kOuterTrn);
  tfOuterTrn = any(bigPartsOuterTrn,2);
  assert(all(tfOuterTst+tfOuterTrn==1));
  
  tblOuterTst = ttrn(tfOuterTst,MFTable.FLDSID);
  tblOuterTrn = ttrn(tfOuterTrn,MFTable.FLDSID);
  innerparts = bigPartsOuterTrn(tfOuterTrn,:);
  assert(size(innerparts,2)==6);
  
  innerSplit = innerparts(:,[1 3 5]) + innerparts(:,[2 4 6]);
  innerSplit = logical(innerSplit);
  
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  assert(isempty(intersect(tblOuterTst,tblOuterTrn)));
  szassert(innerSplit,[nOuterTrn 3]);
  assert(all(sum(innerSplit,2)==1));
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));  
  gmseOuterTrnC = gmse(tfOuterTrn);
  for kinner=1:3
    fprintf(1,'Inner split %d:\n',kinner);
    sortedsummary(gmseOuterTrnC(innerSplit(:,kinner)))
  end
  
  fnameTblOuterTst = sprintf('outer3_easy_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('outer3_easy_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('outer3_easy_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end

%% OUTER: HARD
%% INNER: HARD

% For "hard" we do a strict partitioning by mouse. pretty rough, some of
% the 9-fold splits are just one mouse

smse = stratifiedGroupSplit(BIGKFOLD,ones(size(gmse)),gmse,...
  'shufflePartCats',true);
tftestmsesplit = arrayfun(@(x)smse==x,1:BIGKFOLD,'uni',0);
tftestmsesplit = cat(2,tftestmsesplit{:});

bigparts = tftestmsesplit;
szassert(bigparts,[n BIGKFOLD]); % col i is indicator vec for part i
unique(sum(bigparts,2))
sum(bigparts,1)
for k=1:BIGKFOLD
  tfk = bigparts(:,k);
  fprintf(1,'Fold %d\n',k);
  sortedsummary(gmse(tfk))
end
%%

for iOuterTstSplit=1:3
  kOuterTst = (1:3) + (iOuterTstSplit-1)*3;
  kOuterTrn = setdiff(1:9,kOuterTst);
  tfOuterTst = any(bigparts(:,kOuterTst),2);
  bigPartsOuterTrn = bigparts(:,kOuterTrn);
  tfOuterTrn = any(bigPartsOuterTrn,2);
  assert(all(tfOuterTst+tfOuterTrn==1));
  
  tblOuterTst = ttrn(tfOuterTst,MFTable.FLDSID);
  tblOuterTrn = ttrn(tfOuterTrn,MFTable.FLDSID);
  innerparts = bigPartsOuterTrn(tfOuterTrn,:);
  assert(size(innerparts,2)==6);
  
  p6 = randperm(6);
  innerSplit = innerparts(:,p6(1:3)) + innerparts(:,p6(4:6));
  innerSplit = logical(innerSplit);
  
  nOuterTst = height(tblOuterTst);
  nOuterTrn = height(tblOuterTrn);
  assert(isempty(intersect(tblOuterTst,tblOuterTrn)));
  szassert(innerSplit,[nOuterTrn 3]);
  assert(all(sum(innerSplit,2)==1));
  
  fprintf('Outer Tst Fold %d. nOuterTst/Trn: %d/%d. inner splits: %s\n',...
    iOuterTstSplit,nOuterTst,nOuterTrn,mat2str(sum(innerSplit,1)));  
  gmseOuterTrnC = gmse(tfOuterTrn);
  for kinner=1:3
    fprintf(1,'Inner split %d:\n',kinner);
    sortedsummary(gmseOuterTrnC(innerSplit(:,kinner)))
  end
  
  fnameTblOuterTst = sprintf('outer3_hard_fold%02d_tblTst.mat',iOuterTstSplit);
  fnameTblOuterTrn = sprintf('outer3_hard_fold%02d_tblTrn.mat',iOuterTstSplit);
  fnameInner = sprintf('outer3_hard_fold%02d_inner3.mat',iOuterTstSplit);
  save(fnameTblOuterTst,'tblOuterTst');
  save(fnameTblOuterTrn,'tblOuterTrn');
  save(fnameInner,'innerSplit');
end
