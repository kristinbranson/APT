%% Orig "truncated lbl" file from Mayank
%
% note, lbl.labeledposMarked doesn't have expected number of els compared
% to .labeledpos and .movieFilesAll. So we ignore this field entirely as we 
% cannot even know the correspondence of labeledposMarked to the other
% fields.
% 
lbl = load('f:\aptStephenCPRInvestigate20180327\FlyHeadStephenRound2_Janelia.lbl','-mat');
mfa = lbl.movieFilesAll;
nMov = size(mfa,1);

%% Find dups
[dupcats,idupcats] = finddups(mfa(:,1));
[dupcats2,idupcats2] = finddups(mfa);
isequal(dupcats,dupcats2)
isequaln(idupcats,idupcats2)

%% Dup analysis
iMovRm = [];
for i=1:numel(dupcats)
  fprintf('dupcat %d contains %d movs.\n',i,numel(dupcats{i}));
  lposes = lbl.labeledpos(dupcats{i});
  tflposSame = isequaln(lposes{:});  
  if tflposSame
    fprintf('...lposes all match. adding %d movs to rm list.\n',numel(dupcats{i})-1);
    iMovRm = [iMovRm dupcats{i}(2:end)];
  else
    assert(numel(dupcats{i})==2); % empirically found
    fLbled1 = frameLabeled(lposes{1});
    fLbled2 = frameLabeled(lposes{2});
    assert(isempty(fLbled1)); % empirically found
    assert(nnz(~isnan(lposes{1}))==0);
    assert(~isempty(fLbled2)); % empirically found
    assert(nnz(~isnan(lposes{2}))>0);
    fprintf('...lposes do NOT match. 1st has no labels. keeping 2nd.\n');
    iMovRm = [iMovRm dupcats{i}(1)];
    
%     fprintf('..... 1st mov has labeled frms: %s.\n',mat2str(fLbled1));
%     fprintf('..... 2nd mov has labeled frms: %s.\n',mat2str(fLbled2));
  end
end

assert(numel(iMovRm)==numel(unique(iMovRm)));
fprintf('Total rows to remove: %d\n',numel(iMovRm));
lbl.labeledpos(iMovRm,:) = [];
lbl.movieFilesAll(iMovRm,:) = [];
lbl = rmfield(lbl,'labeledposMarked');

save FlyHeadStephenRound2_Janelia_ALCLEANED_20180412.lbl -struct lbl;

%%
lbl = load('FlyHeadStephenRound2_Janelia_ALCLEANED_20180412.lbl','-mat');
mfa = lbl.movieFilesAll;
nMov = numel(lbl.labeledpos);

%%
s = struct(...
  'movFile',cell(0,1),...
  'flyID',[],...
  'frm',[],...
  'pLbl',[]);

for iMov=1:nMov
  movFiles = mfa(iMov,:);
  lpos = lbl.labeledpos{iMov};

  %movFiles = regexprep(movFiles,'/groups/huston/hustonlab','Z:'); % unnec prob
  flyid = parseSHfullmovie(movFiles{1});

  fLbled = frameLabeled(lpos);
  nf = numel(fLbled);
  pLbl = lpos(:,:,fLbled,:);
  szassert(pLbl,[10 2 nf]);
  pLbl = reshape(pLbl,[20 nf])';

  for iF=1:nf
    s(end+1,1).movFile = movFiles;
    s(end).flyID = flyid;
    s(end).frm = fLbled(iF);
    s(end).pLbl = pLbl(iF,:);
  end
end

tblMayank = struct2table(s);

%% OLD STUFF
%tblMayank.movFile = regexprep(tblMayank.movFile,'Z:/flp-chrimson_experiments','flp-chrimson_experiments');


%% Mayank massage
[~,tblMayank.movID] = cellfun(@parseSHfullmovie,tblMayank.movFile(:,1),'uni',0);
[~,tblMayank.movID2] = cellfun(@parseSHfullmovie,tblMayank.movFile(:,2),'uni',0);
tblMayank = tblMayank(:,{'movFile' 'movID' 'movID2' 'flyID' 'frm' 'pLbl'});
tblMayank = sortrows(tblMayank,{'flyID' 'movID' 'frm'});

nowstr = datestr(now,'yyyymmddTHHMMSS');
fname = sprintf('tblMayank_%s.mat',nowstr);
save(fname,'tblMayank');
fname = sprintf('tblMayank_%s.csv',nowstr);
writetable(tblMayank,fname);

%% dup anls 2
tM = load('tblMayank_20180412T190459.mat');
tM = tM.tblMayank;
%% lbl dups
[dupcats,idupcats] = finddups(tM.pLbl,'verbose',true);
%%
rowIDs = strcat(tM.movID,'#',strtrim(cellstr(num2str(tM.flyID))),'#',strtrim(cellstr(num2str(tM.frm))));
[dupcats,idupcats] = finddups(rowIDs,'verbose',true); % EMP: no dups found

%% pLbl dup confirmation.
tMdup = load('tblMayankDupExample.mat');
tMdup = tMdup.tMdupexample;
lbl = load('f:\aptStephenCPRInvestigate20180327\FlyHeadStephenRound2_Janelia.lbl','-mat');
iMov = find(strcmp(lbl.movieFilesAll(:,1),tMdup.movFile{1}))
lposOrig = lbl.labeledpos{iMov};
save tblMayankDupExample.mat -append lposOrig

%% SH says pLbl dups are ok. browse them
% Emp: 8 pairs of dup frames where movs are different but labels/frmnumbers
% are identical. Weird, but only 8 pairs so leave them.
for i=1:numel(dupcats)
  idx = dupcats{i};
  tMdups = tM(idx,{'lblCat' 'movFile' 'movID' 'frm'});    
  if isscalar(unique(tMdups.movID)) && isequal(tMdups.frm,(tMdups.frm(1):tMdups.frm(end))')
    % none; expected case
  else
    disp(tMdups)
  end
end

% EMP: 536 rows are in a dupcat, ie nnz(~isnan(idupcats))==536.