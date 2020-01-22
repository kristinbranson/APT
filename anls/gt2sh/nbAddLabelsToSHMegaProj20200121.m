%%
dd = dir('BenDec2019/*.lbl');
lblsnew = {dd.name}';
lblsnew = fullfile('BenDec2019',lblsnew);
nlblsnew = numel(lblsnew);
%% get all new movs
mfa = [];
tlbl = [];
tlblconc = [];
currmovoffset = 0;
for i=1:nlblsnew
  ldnew = loadLbl(lblsnew{i});
  mfacurr = ldnew.movieFilesAll;
  mfacurr = regexprep(mfacurr,'Z:\','/groups/huston/hustonlab/');
  mfacurr = FSPath.standardPath(mfacurr);
  
  assert(currmovoffset==size(mfa,1)); % offset is equal to number of movsets accumulated
  tlblcurr = Labeler.lblFileGetLabels(ldnew);

  tlblcurrconc = tlblcurr;
  tlblcurrconc.mov = mfacurr(tlblcurrconc.mov);
  
  [imovhaslbls,~,ic] = unique(tlblcurr.mov);
  mfacurr = mfacurr(imovhaslbls,:);
  tlblcurr.mov = ic + currmovoffset; 
    
  mfa = cat(1,mfa,mfacurr);
  tlbl = cat(1,tlbl,tlblcurr);
  currmovoffset = currmovoffset + size(mfacurr,1);
  tlblconc = cat(1,tlblconc,tlblcurrconc);
end
tlbl.tfocc = logical(tlbl.tfocc);
tlblconc.tfocc = logical(tlblconc.tfocc);
SAVEFILE = 'newmovslbls_20200121.mat';
save(SAVEFILE,'-append','mfa','tlbl','tlblconc');
fprintf('Saved %s\n',SAVEFILE);

%% unique movsets
movsetID = MFTable.formMultiMovieIDArray(mfa);
[~,ia] = unique(movsetID);
mfaun = mfa(ia,:);
% all rows unique

%% parse for flynames
addpath /groups/branson/home/leea30/git/apt.winter2019/anls/gt2sh
%%
[flyid,trial,movpath] = cellfun(@parseSHfullmovie,mfa,'uni',0);
flyid = cell2mat(flyid);
szassert(flyid,size(mfa));
assert(isequal(flyid(:,1),flyid(:,2)));
flyid = flyid(:,1);

%% get calibs
FLY2CALIB = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';
fly2calib = readtable(FLY2CALIB,'ReadVariableNames',false,'Delimiter',',');
fly2calib.Properties.VariableNames= {'fly' 'calib'};
%fly2calib.calib = FSPath.standardPath(regexprep(fly2calib.calib,'Z:\','/groups/huston/hustonlab'));
[tf,loc] = ismember(flyid,fly2calib.fly);
assert(all(tf));
calibs = fly2calib.calib(loc);

save(SAVEFILE,'-append','calibs');

%% get crops
ALLCROPS = '/groups/branson/home/leea30/sh/addlbls20200121/crop_locs_all.mat';
crops = load(ALLCROPS);
crops = crops.all_crops;
cropflies = cell2mat(crops(:,1));
croprois = cat(3,crops(:,2),crops(:,3));
%tcrop = table(cell2mat(crops(:,1)),rois,'VariableNames',{'fly' 'crops'});
[tf,loc] = ismember(flyid,cropflies);
assert(all(tf));
crois = croprois(loc,:,:);
tf = cellfun(@isempty,crois);
assert(nnz(tf)==0);
crois = cell2mat(crois);
size(flyid)
size(crois)

crops = crois;

save(SAVEFILE,'-append','crops');

%% 
APTDATA = '/groups/branson/bransonlab/apt/experiments/data/';
PROJ = {
  'sh_trn4992_gtcomplete_cacheddata_updated20190402.lbl';
  'sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402.lbl';
  };
projFull = fullfile(APTDATA,PROJ);
STRIP = {
  'sh_trn4992_gtcomplete_cacheddata_updated20190402_dlstripped.lbl'
  'sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'
  };
stripFull = fullfile(APTDATA,STRIP);
%%
stuff = load('newmovslbls_20200121.mat');
createProject(projFull{2},stuff.mfa,...
  'clearBaseProj',false,...
  'tblLbls',stuff.tlbl(:,MFTable.FLDSCORE),...
  'calibFiles',stuff.calibs,...
  'cropRois',stuff.crops,...
  'outfile',fullfile(pwd,'sh_trn5017_20200121.lbl'),...
  'diaryfile',fullfile(pwd,'dry_sh_trn5017_20200121.txt'));

%% compare to prev

s1 = load('sh_trn5017_20200121_stripped.lbl','-mat');
s0 = load('../../blapt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl','-mat');
% mov causes trouble with reads
mov = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_380_to_fly_389_12_9_16_SS02323_and_SS002323POsilenced/SS002323CsChrimson_lexA48A07Kir/fly382/C001H001S0001/C001H001S0001_c.avi';
imov = find(strcmp(s0.movieFilesAll(:,1),mov));
%%
isequaln(s0.movieFilesAll,s1.movieFilesAll(1:size(s0.movieFilesAll,1),:))
FLDS = {
  'preProcData_I'
  'preProcData_MD_frm'
  'preProcData_MD_iTgt'
  'preProcData_MD_mov'
  'preProcData_MD_nNborMask'
  'preProcData_MD_roi'
  'preProcData_MD_tfocc'
  'preProcData_P'
  };
for f=FLDS(:)',f=f{1};
  tf = isequaln(s0.(f),s1.(f)(1:4992,:));
  fprintf(1,'%s: %d\n',f,tf);
end
%%
idx = find(s0.preProcData_MD_mov==imov)
frms = s0.preProcData_MD_frm(idx);
nfrm = numel(frms);
roisvw0 = s0.preProcData_MD_roi(idx,1:4)
[rf0,nf0,fid0] = get_readframe_fcn(mov);
[rf1,nf1,fid1] = get_readframe_fcn(mov,'preload',true);
I0 = arrayfun(@(x)rf0(x),frms,'uni',0);
I1 = arrayfun(@(x)rf1(x),frms,'uni',0);
%%
xlo = roisvw0(1,1);
xhi = roisvw0(1,2);
ylo = roisvw0(1,3);
yhi = roisvw0(1,4);
I0crop = cellfun(@(x)x(ylo:yhi,xlo:xhi,1),I0,'uni',0);
I1crop = cellfun(@(x)x(ylo:yhi,xlo:xhi,1),I1,'uni',0);
I0stripped = s0.preProcData_I(idx,1);
I1stripped = s1.preProcData_I(idx,1);
isequal(I0crop,I1stripped)
isequal(I1crop,I1stripped)
%%
d = nan(size(I0crop));
for i=1:numel(I0crop)
  tmp = I0stripped{i}-I1stripped{i};
  d(i) = max(abs(tmp(:)));
end
%%
ld = cellfun(@loadLbl,projFull);
s = cellfun(@(x)load(x,'-mat'),stripFull);