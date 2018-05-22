%%
%load trnDataSH_Apr18.mat;
load trnData20180503.mat

%%
t = tMain20180503;
id = strcat(t.lblCat,'#',numarr2trimcellstr(t.flyID),'#',t.movID);
idC = categorical(id);
tCats = sortedsummary(idC)
idC = reordercats(idC,tCats.cats);
%%
hFig = figure;
cla;
ax = createsubplots(2,1);
plot(ax(1),tCats.cnts,'.');
grid(ax(1),'on');
plot(ax(2),tCats.cnts,'o');
grid(ax(2),'on');
set(ax(2),'Xscale','log','YScale','log');
%%
npart = 50;
c = cvpartition(idC,'kfold',npart);
parts = arrayfun(@(z)test(c,z),1:npart,'uni',0);
parts = cat(2,parts{:});
szassert(parts,[height(t) npart]); % col i is indicator vec for part i
partsum = sum(parts,2);
unique(partsum)

%% 20180508. New 3-fold splits
% 1. Easy split, as before
% 2. Harder split. Balance lblCat, but don't have intersecting flies

NFOLD = 3;
t = tMain20180503;
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
xvMain3Hard = false(height(t),NFOLD);
for ifold=1:NFOLD
  xvMain3Hard(:,ifold) = s==ifold;
end
sum(xvMain3Hard,1)
unique(sum(xvMain3Hard,2))
save trnSplit20180509.mat xvMain3Hard

%% easy 
c = cvpartition(idC,'kfold',NFOLD);
parts = arrayfun(@(z)test(c,z),1:NFOLD,'uni',0);
parts = cat(2,parts{:});
szassert(parts,[height(t) NFOLD]); % col i is indicator vec for part i
unique(sum(parts,2))
sum(parts,1)

xvMain3Easy = parts;
save trnSplit20180509.mat -append xvMain3Easy;



%% INTERRUPT 20180423, generate 3-fold, 5-fold xv sets
npart = 5;
c = cvpartition(idC,'kfold',npart);
parts = arrayfun(@(z)test(c,z),1:npart,'uni',0);
parts = cat(2,parts{:});
szassert(parts,[height(t) npart]); % col i is indicator vec for part i
unique(sum(parts,2))
sum(parts,1)

xvFRsplit5 = parts;
save trnSplits_20180418T173507.mat -append xvFRsplit5;

%% INTERRUPT 20180420, generate a ntrn=50 training set, not a part of current 100-200-... subset chain
npart = 100;
c = cvpartition(idC,'kfold',npart);
parts = arrayfun(@(z)test(c,z),1:npart,'uni',0);
parts = cat(2,parts{:});
szassert(parts,[height(t) npart]); % col i is indicator vec for part i
partsum = sum(parts,2);
unique(partsum)

parts50 = parts(:,[2:end 1]);
trnSet50 = parts50(:,1);
save trnSplits_20180418T173507.mat -append parts50 trnSet50

%% EMP: reorder cols of parts so the first N cols all have 100 (vs 99)
parts = parts(:,[2:end 1]);
sum(parts,1)

%%
NP = [1 2 4 8 16 32 50];
trnSets = cell(numel(NP),1);
trnSetCatCnts = cell(numel(NP),1);
for i=1:numel(NP)
  npartsinc = NP(i);
  indvec = any(parts(:,1:npartsinc),2);
  trnSets{i} = indvec;
  idCthis = idC(indvec);
  trnSetCatCnts{i} = countcats(idCthis);
end
trnSets = cat(2,trnSets{:}); % ith col is ith training set
trnSetCatCnts = cat(2,trnSetCatCnts{:});
%%
figure;
ax = createsubplots(2,1);
plot(ax(1),tCats.cnts,'.');
hold(ax(1),'on');
plot(ax(1),trnSetCatCnts);
grid(ax(1),'on');
plot(ax(2),tCats.cnts,'o');
hold(ax(2),'on');
plot(ax(2),trnSetCatCnts);
grid(ax(2),'on');
set(ax(2),'Xscale','log','YScale','log');

%%
nowstr = datestr(now,'yyyymmddTHHMMSS');
fname = sprintf('trnSplits_%s.mat',nowstr);
save(fname,'idC','npart','parts','NP','trnSets');


