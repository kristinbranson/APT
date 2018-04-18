%% OBJECTIVE: Merge SH with Mayank data so we can use Mayank paths

%%
tM = load('tblMayank_20180412T190459.mat');
tM = tM.tblMayank;
t = load('tblSH_idDupsRemoved_20180415T095958.mat');
t = t.tIDdupsRmed;

tM.movFile = FSPath.standardPath(strtrim(tM.movFile));
tM.movID = FSPath.standardPath(strtrim(tM.movID));
tM.movID2 = FSPath.standardPath(strtrim(tM.movID2));

t.movFile = FSPath.standardPath(strtrim(t.movFile));
t.movID = FSPath.standardPath(strtrim(t.movID));
t.movID2 = FSPath.standardPath(strtrim(t.movID2));

%% Add 'trial' to tables
[~,trl1] = cellfun(@parseSHfullmovie,t.movFile(:,1));
[~,trl2] = cellfun(@parseSHfullmovie,t.movFile(:,2));
idx = find(trl1~=trl2);
fprintf(1,'SH: Removing %d rows where trials don''t match.\n',numel(idx));
t.trial = trl1;
t(idx,:) = [];

[~,trl1] = cellfun(@parseSHfullmovie,tM.movFile(:,1));
[~,trl2] = cellfun(@parseSHfullmovie,tM.movFile(:,2));
assert(isequal(trl1,trl2));
tM.trial = trl1;

%%
n = height(t);
nM = height(tM);

%% tblID reduced dupcheck 
ids = strcat(t.movID,'#',numarr2trimcellstr(t.frm));
idsM = strcat(tM.movID,'#',numarr2trimcellstr(tM.frm));
ids2 = strcat(numarr2trimcellstr(t.flyID),'#',numarr2trimcellstr(t.trial),'#',numarr2trimcellstr(t.frm));
ids2M = strcat(numarr2trimcellstr(tM.flyID),'#',numarr2trimcellstr(tM.trial),'#',numarr2trimcellstr(tM.frm));
save tmpIDS.mat ids idsM ids2 ids2M;

t.id = ids;
t.id2_nonunique = ids2;
tM.id = idsM;
tM.id2_nonunique = ids2M;
%%
t = t(:,{'lblCat' 'lblFile' 'iMov' 'movFile' 'movID' 'movID2' 'flyID' 'trial' 'frm' 'id' 'id2_nonunique' 'pLbl' 'pLblDate'});
tM = tM(:,{'movFile' 'movID' 'movID2' 'flyID' 'trial' 'frm' 'id' 'id2_nonunique' 'pLbl'});

%%
[dupcats,idupcats] = finddups(ids,'verbose',true); 
[dupcatsM,idupcatsM] = finddups(idsM,'verbose',true);
[dupcats2,idupcats2] = finddups(ids2,'verbose',true); % EMP: ids2 not a sound ID, it's not unique u need the movie too
[dupcats2M,idupcats2M] = finddups(ids2M,'verbose',true);

%% Reconcile round 1. use ids/idsM (movID,frm)

% idxM2S(iM) is:
% * idxIntoStephen, if there is a unique corresponding SH row.
% * 0, if there is no corresponding SH row 
idxM2S = zeros(nM,1);
tfMbad = false(nM,1); % flag indicating an M row is suspect 
tfSbad = false(n,1); % flag indicating an SH row is suspect 

fprintf('Stephen: %d rows. Mayank: %d rows\n',n,nM);

[tf,loc] = ismember(idsM,ids);
idxM = find(tf);
idxS = loc(tf);
fprintf('Mayank in Stephen: %d/%d rows\n',numel(idxM),nM);
pM = tM(idxM,:).pLbl;
pS = t(idxS,:).pLbl;
tfLblMismatch = arrayfun(@(x)~isequaln(pM(x,:),pS(x,:)),(1:numel(idxM))');

idxM2S(idxM(~tfLblMismatch)) = idxS(~tfLblMismatch);
tfMbad(idxM(tfLblMismatch)) = true;
tfSbad(idxS(tfLblMismatch)) = true;

tfMremain = idxM2S==0 & ~tfMbad;

fprintf('  %d/%d potential matches. %d successful matches.\n',...
  nnz(tf),numel(tf),nnz(~tfLblMismatch));
fprintf('  %d rows marked as bad. %d/%d M rows remain.\n',...
  nnz(tfLblMismatch),nnz(tfMremain),numel(tfMremain));

%% Reconcile round 2. use ids2/idsM2 (flyID,trial,frm) AND matching pLbls

[tf,loc] = ismember(ids2M,ids2);
tf(idxM2S>0 | tfMbad) = false; % don't match previously matched, or bad
loc(idxM2S>0 | tfMbad) = 0; % etc

idxM = find(tf);
idxS = loc(tf);
fprintf('Mayank in Stephen: %d/%d rows\n',numel(idxM),nM);
pM = tM(idxM,:).pLbl;
pS = t(idxS,:).pLbl;
tfLblMismatch = arrayfun(@(x)~isequaln(pM(x,:),pS(x,:)),(1:numel(idxM))');

idxM(tfLblMismatch) = [];
idxS(tfLblMismatch) = [];

idxM2S(idxM) = idxS;
tfMremain = idxM2S==0 & ~tfMbad;

fprintf('  %d new matches. %d remain.\n',numel(idxM),nnz(tfMremain));

% EMP: nnz(tfMbad)==0 and nnz(tfSbad)==0

%% Final/joined reconciled table. 
tMnoMatchWithS = tM(idxM2S==0,:);
tMtoJoin = tM(idxM2S>0,:);
flds = tblflds(tMtoJoin);
tMtoJoin.Properties.VariableNames = strcat(flds,'_m');
tStoJoin = t(idxM2S(idxM2S>0),:);

tFinalReconciled = [tStoJoin tMtoJoin];

%%
tfGT = ismember(t.lblCat,{'bAxis' '2016'}) & ~ismember(t.flyID,tM.flyID); % second clause could be weaked to ~ismember(t.flyID,tFinalReconciled.flyID)
tGT = t(tfGT,:);

tfSNotMatchedAndNonGT = true(n,1);
tfSNotMatchedAndNonGT(idxM2S(idxM2S>0)) = false;
tfSNotMatchedAndNonGT(tfGT) = false;
tSNotMatchedAndNonGT = t(tfSNotMatchedAndNonGT,:);

fprintf(1,'height tFinal/tMnoMatch/tGT/tSnotMatched: %d/%d/%d/%d\n',...
  height(tFinalReconciled),height(tMnoMatchWithS),height(tGT),height(tSNotMatchedAndNonGT));

nowstr = datestr(now,'yyyymmddTHHMMSS');
savefile = sprintf('tblFinalReconciled_%s',nowstr);
save(savefile,'tFinalReconciled','tMnoMatchWithS','tGT','tSNotMatchedAndNonGT');

%% checks
isequaln(tFinalReconciled.pLbl,tFinalReconciled.pLbl_m)
all(strcmp(tFinalReconciled.id,tFinalReconciled.id_m) | ...
  strcmp(tFinalReconciled.id2_nonunique,tFinalReconciled.id2_nonunique_m))
tf = strcmp(tFinalReconciled.id,tFinalReconciled.id_m);
tFinalReconciled(~tf,{'id' 'id_m'})

summary(categorical(tFinalReconciled.lblCat))
summary(categorical(tSNotMatchedAndNonGT.lblCat))

tfNaNLbl = any(isnan(tFinalReconciled.pLbl),2);
nnz(tfNaNLbl)
