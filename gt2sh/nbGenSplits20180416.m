%%
load tblFinalReconciled_20180415T212437;
%%
id = strcat(t.lblCat,'#',numarr2trimcellstr(t.flyID),'#',t.movID);
idC = categorical(id);
tCats = sortedsummary(idC)
idC = reordercats(idC,tCats.cats);
%%
hFig = figure;
cla;
ax = createsubplots(2,1);
plot(ax(1),tCats.idCcnts,'.');
grid(ax(1),'on');
plot(ax(2),tCats.idCcnts,'o');
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

