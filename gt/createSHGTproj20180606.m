%%
SELECTEDFRAMES = '/groups/branson/bransonlab/apt/experiments/data/SelectedGTFrames_SJH_20180603.mat';
sel = load(SELECTEDFRAMES);
t = sel.frames2label;

%%
FLY2CALIB = '/groups/branson/bransonlab/apt/experiments/data/fly2DLT_lookupTableAL.csv';
fly2cal = readtable(FLY2CALIB);
[tf,loc] = ismember(t.flyID,fly2cal.fly);
assert(all(tf));
t.calibFile = fly2cal.calibfile(loc);

%% 
tfIntra = strcmp(t.type,'intra');
movsIntra = unique(t(tfIntra,:).movFile(:));
mMov2IntraMov = containers.Map('KeyType','char','ValueType','char');
nMovsIntra = numel(movsIntra);
for i=1:nMovsIntra
  [movP,movF,movE] = fileparts(movsIntra{i});
  mov2F = [movF '_i'];
  cd(movP);  
  cmd = sprintf('ln -s %s %s',[movF movE],[mov2F movE]);
  fprintf('in %s. %s\n',pwd,cmd);  
  %system(cmd);
  
  mMov2IntraMov(movsIntra{i}) = fullfile(movP,[mov2F movE]);
end

%%
% movFile_i = cell(size(t.movFile));
% movFile_i(:) = {''};
for i=1:height(t)
  if tfIntra(i)
    t.movFile{i,1} = mMov2IntraMov(t.movFile{i,1});
    t.movFile{i,2} = mMov2IntraMov(t.movFile{i,2});    
  end
end


%% generate list of all unique movFile pairs and all unique movFile_i pairs.
movID = strcat(t.movFile(:,1),'#',t.movFile(:,2));
[~,idx] = unique(movID);
movsUn = t.movFile(idx,:);
nMovSet = size(movsUn,1);
movsUn = movsUn(randperm(nMovSet),:);
%lObj.movieSetAdd(movsUn(1,:));


%%
lObj = Labeler;
lObj.projLoad('/groups/branson/home/leea30/apt/gtsh/APTProject.lbl');
lObj.gtSetGTMode(true);

%% add the movies/viewcals
diary dryGTmovsAdd.txt
for i=1:nMovSet
  mov1 = movsUn{i,1};
  mov2 = movsUn{i,2};
  tf1 = strcmp(mov1,t.movFile(:,1));
  tf2 = strcmp(mov2,t.movFile(:,2));
  assert(isequal(tf1,tf2));
  calfile = unique(strtrim(t.calibFile(tf1)));  
  assert(isscalar(calfile));

  calfile = calfile{1};
  [crObj,tfSetViewSizes] = CalRig.loadCreateCalRigObjFromFile(calfile);

  lObj.movieSetAdd({mov1 mov2},'offerMacroization',false);
  lObj.movieSet(lObj.nmoviesGT);
  lObj.viewCalSetCurrMovie(crObj,'tfSetViewSizes',tfSetViewSizes);

  drawnow;
  fprintf(1,'%d: %s. %d\n',i,class(crObj),tfSetViewSizes);
end
diary off

%% save it

%% Check diary.

%% macroize
lObj.movieFilesMacroize('/groups/huston/hustonlab/flp-chrimson_experiments','flpCE');

%% check MFAF, calibs

%% gtSugg

% make gtSuggMFT
tGT = t(:,{'movFile' 'frm'});
tGT.iTgt = ones(height(t),1);
tblflds(tGT)

tGT.Properties.VariableNames = {'mov' 'frm' 'iTgt'};
mfaf = lObj.movieFilesAllGTFull;
[tf1,loc1] = ismember(tGT.mov(:,1),mfaf(:,1));
[tf2,loc2] = ismember(tGT.mov(:,2),mfaf(:,2));
isequal(tf1,tf2)
isequal(loc1,loc2)
tGT.movFull = tGT.mov;
tGT.mov = MovieIndex(-loc1);

tGTsugg = tGT;
save tblSHgt_withintra_20180605.mat -append tGTsugg;

lObj.gtSetUserSuggestions(tGTsugg(:,MFTable.FLDSID),'sortcanonical',true);

%% SH try labeling

%% double-check, check SH labels

lbl = load('gtsh_main_1150_v1_20180605_SJHcopy_050618_1731.lbl','-mat');
kb = load('SelectedGTFrames_SJH_20180603.mat');
%%
tGT = kb.frames2label;
tfIntra = strcmp(tGT.type,'intra');
nIntra = nnz(tfIntra);
iIntra = find(tfIntra);
intraOK = false(nIntra,1);
for jIntra=1:nIntra
  i = iIntra(jIntra);
  tGTrow = tGT(i,:);
  tf = strcmp(tGTrow.movFile(:,1),tGT.movFile(:,1)) & ...
       strcmp(tGTrow.movFile(:,2),tGT.movFile(:,2)) & ...
       tGTrow.frm==tGT.frm;
  typeVals = tGT(tf,:).type
  assert(nnz(tf)==2);
  assert(nnz(strcmp(typeVals,'intra'))==1);
end

%%
tGT(tfIntra,:).movFile = cellfun(@intraizeMovie,tGT(tfIntra,:).movFile,'uni',0);
tGT.movFile = regexprep(tGT.movFile,'\\','/');
tGT = tGT(:,{'movFile' 'frm'});
tGT.Properties.VariableNames = {'mov' 'frm'};
tGT.iTgt = ones(height(tGT),1);
tGT = sortrows(tGT,{'mov' 'frm'},{'ascend' 'ascend'});

%%
tLbl = lbl.gtSuggMFTable;
mfagt = lbl.movieFilesAllGT;
iMov = abs(tLbl.mov);
movfiles = mfagt(iMov,:);
movfiles = regexprep(movfiles,'\$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
tLbl.mov = movfiles;
tLbl = sortrows(tLbl,{'mov' 'frm'},{'ascend' 'ascend'});

isequal(tGT,tLbl)

%%
fly2dlt = readtable('z:/apt/experiments/data/fly2DLT_lookupTableAL_win.csv','Delimiter',',');
mfafgt = lbl.movieFilesAllGT;
mfafgt = regexprep(mfafgt,'\$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
nmfaf = size(mfafgt,1);
for i=1:nmfaf
  tf = strcmp(mfafgt{i,1},tGT.movFile(:,1));  
  flyID = unique(tGT(tf,:).flyID);
  assert(isscalar(flyID));
  iCalib = find(flyID==fly2dlt.fly);
  assert(isscalar(iCalib));
  calfile = fly2dlt.calibfile{iCalib};
  calfile = regexprep(calfile,'Z:/','\\\\dm11.hhmi.org\\hustonlab\\');
  cal1 = lbl.viewCalibrationDataGT{i};
  cal2 = CalRig.loadCreateCalRigObjFromFile(calfile);
  if isa(cal1,'CalRigSH')
    assert(isequal(cal1.kineData,cal2.kineData));
  elseif isa(cal1,'OrthoCamCalPair')
    assert(isequal(cal1.rvecs,cal2.rvecs));
  end
  disp(i);
end

%%
hFig = figure;
ax = axes;
hold(ax,'on');
lposgt = lbl.labeledposGT;
for i=1:numel(lposgt)
  lpos = lposgt{i};
  lpos = SparseLabelArray.full(lpos);
  f = frameLabeled(lpos);
  nf = numel(f);
  for iF=1:nf
    xy = lpos(:,:,f(iF),1);
    sum(~isnan(xy(:)))
  end
end