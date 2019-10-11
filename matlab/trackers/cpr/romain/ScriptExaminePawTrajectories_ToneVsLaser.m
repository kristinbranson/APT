%% Compare tone-evoked and laser-evoked reaches. 
%
% This script compares the reach trajectories evoked by a tone to those
% evoked by optogenetic inhibition (laser). It aligns trajectories based on
% when the lift and grab behaviors happen, and looks at how well we can
% predict the trial type based on the trajectory. It also plots
% visualizations of these trajectories and their differences. 
%
% Plots:
% * Example successful and unsuccessful grab trajectories for control:
%   -- AllSuccessfulGrabs3D_*trx_*
%   -- AllSuccessfulLift2Grab3D_*trx_*
%   -- AllGrabs_*
% * How well can we predict trial type given trajectory position at various
% time points:
%   -- NearestNeighborTrajClassificationVsChance*
%   -- PositionAlongLift2Grab_*
%
% Code dependencies:
% JAABA http://jaaba.sourceforge.net/ (mine is installed in
% /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect)
% Camera calibration toolbox http://www.vision.caltech.edu/bouguetj/calib_doc/
% (mine is installed in
% /groups/branson/home/bransonk/codepacks/TOOLBOX_calib)
%
% Data locations:
% Paw is assumed to be tracked already, and manually corrected trajectories
% are in the directories specified in trxdirs and trxfilestrs cell. 
% Videos are assumed to be in directories specified in paw trajectory
% files. 
% Manually labeled behavior timings are assumed to be in directories
% specified in ppfiles. 
%
% Mat file with state at end of run:
% data/ToneVsLaserData20150717.mat
%

%% set up paths

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath /groups/branson/home/bransonk/codepacks/TOOLBOX_calib;
rootdir = '/tier2/hantman';

% ppfiles0 = {'/misc/public/Juan2Adam/post_results/M118_CNO_G6.mat'
%   '/misc/public/Juan2Adam/post_results/M119_CNO_G6.mat'
%   '/misc/public/Juan2Adam/post_results/M122_CNO_G6.mat'
%   '/misc/public/Juan2Adam/post_results/M127_CNO_G6.mat'
%   '/misc/public/Juan2Adam/post_results/M130_CNO_G6.mat'};
%   
% ppfiles = {'data/PostProcessed_M118_20141231.mat'
%   'data/PostProcessed_M119_20141231.mat'
%   'data/PostProcessed_M122_20141231.mat'
%   'data/PostProcessed_M127_20141231.mat'
%   'data/PostProcessed_M130_20150101.mat'};
%
% trxfiles0 = {
%   '/misc/public/Juan2Adam/results/results_M118f.mat'
%   '/misc/public/Juan2Adam/results/results_M119B.mat'
%   '/misc/public/Juan2Adam/results/results_M122b.mat'
%   '/misc/public/Juan2Adam/results/results_M127a.mat'
%   '/misc/public/Juan2Adam/results/results_M130c.mat'
% };

% for i = 1:numel(trxfiles0),
%   mouse = regexp(trxfiles0{i},'results_(M\d+)[^\d]','once','tokens');
%   mouse = mouse{1};
%   tmp = dir(trxfiles0{i});
%   assert(~isempty(tmp));
%   outfile = fullfile('data',sprintf('FixedTrackingResults_%s_%s.mat',mouse,datestr(tmp.date,'yyyymmdd')));
%   if exist(outfile,'file'),
%     tmp2 = dir(outfile);
%     assert(tmp2.bytes==tmp.bytes);
%   else
%     %error('!');
%     unix(sprintf('cp %s %s',trxfiles0{i},outfile));
%   end
%   fprintf('''%s''\n',outfile);
% end

% trxdirs0 = {'/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M134New_20150531'
%   '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M173_20150531'
%   '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M174New_20150531'
%   '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M135New_20150531'
%   '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150329'
%   '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M147_20150615'};
trxdirs = {'/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M134New_20150531/corrected'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M173_20150531/corrected'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M174New_20150531/corrected'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M135New_20150531'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150329/corrected'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M147_20150615/corrected'};
trxfilestrs = {'TrackingResults_M134C3VGATXChR2*.mat'
  'TrackingResults_M173VGATXChR2*.mat'
  'TrackingResults_M174VGATXChR2*.mat'
  'TrackingResults_M135C4VGATXChR2*.mat'
  'TrackingResults_M147VGATXChrR2*.mat'
  'TrackingResults*.mat'};

trxfiles = [];
% trxfiles0 = [];
for i = 1:numel(trxdirs),
  trxdir = trxdirs{i};
  %trxdir0 = trxdirs0{i};
  trxfilestr = trxfilestrs{i};
  trxfiles = [trxfiles,mydir(fullfile(trxdir,trxfilestr))];
%   trxfiles0 = [trxfiles0,mydir(fullfile(trxdir0,trxfilestr))];
end

ppfiles = {'/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M134New_20150531/corrected/M134_all20141203to20150507.mat'
  '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/data/M134_20150427to20150507_SingleLaser.mat'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M173_20150531/corrected/M173_20150415to512_manual.mat'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M174New_20150531/corrected/M174_manual_20150409to20150521.mat'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M135New_20150531/M135_20141204to20150202.mat'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M135New_20150531/M135_20150209to20150506.mat'
  '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/data/M135_20150427to20150506_SingleLaser.mat'
  '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/data/M147_20141208to20150506.mat'};
  
  %mydir('data/PostProcessed_M174/*.mat')];

% laserinfofiles = {'data/M134LaserTimingInfo.csv'
%   'data/M173LaserTimingInfo.csv'
%   'data/M174LaserTimingInfo.csv'};
% laserexpnamecolumn = [4,4,1];
% laseroncolumn = [5,7,2];
% laseroffcolumn = [6,8,3];

exprootdirs = {'/tier2/hantman/Jay/videos/M134C3VGATXChR2'
  '/tier2/hantman/Jay/videos/M173VGATXChR2'
  '/tier2/hantman/Jay/videos/M174VGATXChR2'
  '/tier2/hantman/Jay/videos/M135C4VGATXChR2_anno'
  '/tier2/hantman/Jay/videos/M147VGATXChrR2_anno'};
subdirs_tone = {{'CTR'},{'CTR'},{'CTR'},{'CTR'},{'CTR'}};
subdirs_laser = {{'M1afterCue'},{'CueLaser2sec','CueLaser4sec'},{'CueLaser4sec'},{'M1afterCue'},{'M1afterCue'}};
laserontime_persubdir_tone = {0,0,0,0,0};
laserofftime_persubdir_tone = {2000,[1000,2000],2000,2000,2000};

% expstrs_tone = win2unixpath(expstrs_tone,rootdir);
% assert(all(cellfun(@exist,expstrs_tone)));
% expstrs_laser = win2unixpath(expstrs_laser,rootdir);
% assert(all(cellfun(@exist,expstrs_laser)));

timestamp = now;
savefigdir = sprintf('Trajectories%s',datestr(timestamp,'yyyymmdd'));
if ~exist(savefigdir,'dir'),
  mkdir(savefigdir);
end

moviefilestr = 'movie_comb.avi';

calibrationfile = 'CameraCalibrationParams20150614.mat';
calibrationdata = load(calibrationfile);

uselabeled = true;

%% find all experiments

expstrs_tone = {};
for i = 1:numel(exprootdirs),
  for j = 1:numel(subdirs_tone{i}),
    expstrs_tone = [expstrs_tone,mydir(exprootdirs{i},'name',['^',subdirs_tone{i}{j},'$'],'isdir',true,'recursive',true,'maxdepth',1)]; %#ok<AGROW>
  end
end

expstrs_laser = {};
for i = 1:numel(exprootdirs),
  for j = 1:numel(subdirs_laser{i}),
    expstrs_laser = [expstrs_laser,mydir(exprootdirs{i},'name',['^',subdirs_laser{i}{j},'$'],'isdir',true,'recursive',true,'maxdepth',1)]; %#ok<AGROW>
  end
end

fprintf('Found %d directories containing tone experiments:\n',numel(expstrs_tone));
fprintf('%s\n',expstrs_tone{:});
fprintf('Found %d directories containing laser experiments:\n',numel(expstrs_laser));
fprintf('%s\n',expstrs_laser{:});

%% behavior names and order

% fnsspecial = {'auto_Grab_success','auto_Grab_successtype'};
% fns = fieldnames(rawdata);
% datafns = setdiff(fns(~cellfun(@isempty,regexp(fns,'^auto'))),fnsspecial);
% nstats = numel(datafns);
% 
% % some behaviors don't have counts
% tmp = regexp(datafns,'^auto_([^_]+)_0$','tokens','once');
% behaviors = unique([tmp{:}]);

behaviors = {'Lift','Handopen','Grab','Sup','Atmouth','Chew'};
behaviorfns = {'auto_GS00_Lift_0','auto_GS00_Handopen_0','auto_GS00_Grab_0','auto_GSSS_Sup_0',...
  'auto_GSSS_Atmouth_0','auto_GSSS_Chew_0'};
behaviorfns1 = {'auto_GSSS_Lift_0','auto_GSSS_Handopen_0','auto_GSSS_Grab_0','auto_GSSS_Sup_0',...
  'auto_GSSS_Atmouth_0','auto_GSSS_Chew_0'};
nbehaviors = numel(behaviors);
requiredfns = {'auto_Grab_success','auto_Grab_successtype'};

%% load in trajectory data

trxdata = LoadTrajectoryData(trxfiles,expstrs_laser,expstrs_tone,rootdir,'fixmissing',false);
% trxdata0 = LoadTrajectoryData(trxfiles0,expstrs_laser,expstrs_tone,rootdir);
% 
% expdirs_missing = setdiff({trxdata0.expdir},{trxdata.expdir});
% if ~isempty(expdirs_missing),
%   warning('%d experiments missing in corrected trajectories',numel(expdirs_missing));
%   fprintf('%s\n',expdirs_missing{:});
% end

trxdata0 = trxdata;

tmp = regexp({trxdata.expdir},'/(M\d+)_[^/]*$','tokens','once');
tmp = [tmp{:}];
[mice,~,mouseidx] = unique(tmp);
for mousei = 1:numel(mice),
  fprintf('Mouse %s: %d laser, %d tone experiments.\n',mice{mousei},nnz([trxdata(mouseidx==mousei).islaser]),...
    nnz([trxdata(mouseidx==mousei).istone]));
end

parentdir = cellfun(@fileparts,{trxdata.expdir},'Uni',0);
fprintf('Laser experiments:\n');
[islaser,idxlaser] = ismember(parentdir,expstrs_laser);
for i = 1:numel(expstrs_laser),
  fprintf('%s: %d experiments found\n',expstrs_laser{i},nnz(idxlaser==i));
end
fprintf('Tone experiments:\n');
[istone,idxtone] = ismember(parentdir,expstrs_tone);
for i = 1:numel(expstrs_tone),
  fprintf('%s: %d experiments found\n',expstrs_tone{i},nnz(idxtone==i));
end

%% load in post-processed data

rawdata = [];
for i = 1:numel(ppfiles),
  
  [~,~,ext] = fileparts(ppfiles{i});
  switch ext,
    case '.csv'
      rawdatacurr = ReadRawDataFile(ppfiles{i});
    case '.mat'      
      rawdatacurr = load(ppfiles{i});
      if isfield(rawdatacurr,'mergedData'),
        rawdatacurr = rawdatacurr.mergedData;
      else
        rawdatacurr = rawdatacurr.data;
      end
  end
  expdirs_curr = win2unixpath({rawdatacurr.expfull},rootdir);
  [istone,islaser] = IsToneOrLaserExp(expdirs_curr,expstrs_tone,expstrs_laser);
  dokeep = islaser | istone;
  if ~any(dokeep),
    continue;
  end
  expdirs_curr = expdirs_curr(dokeep);
  rawdatacurr = rawdatacurr(dokeep);
  
  rawmetadata = GetExpTypeFromDir2(expdirs_curr);
  fns = fieldnames(rawmetadata);
  for k = 1:numel(rawdatacurr),
    for j = 1:numel(fns),
      rawdatacurr(k).(fns{j}) = rawmetadata(k).(fns{j});
    end
    rawdatacurr(k).expfull = expdirs_curr{k};
  end
  
  % old code uses "auto" as the name, so replace all labl with auto
  fns = fieldnames(rawdatacurr);
  
  if ~isfield(rawdatacurr,'labl_Lift_0'),
    islabeled = false(1,numel(rawdatacurr));
  else
    islabeled = ~isnan([rawdatacurr.labl_Lift_0]);
  end
  isauto = ~isnan([rawdatacurr.auto_Lift_0]);
  fprintf('%s:\n',ppfiles{i});
  fprintf('%d/%d experiments labeled & not auto\n',nnz(islabeled&~isauto),numel(islabeled));
  fprintf('%d/%d experiments auto & not labeled\n',nnz(~islabeled&isauto),numel(islabeled));
  
  if uselabeled,  
    isauto = ~cellfun(@isempty,regexp(fns,'^auto','once'));
    islabl = ~cellfun(@isempty,regexp(fns,'^labl','once'));
    
    for j = find(islabl(:)'),
      fn0 = fns{j};
      fn1 = strrep(fn0,'labl','auto');
      for k = 1:numel(rawdatacurr),
        rawdatacurr(k).(fn1) = rawdatacurr(k).(fn0);
      end
      rawdatacurr = rmfield(rawdatacurr,fn0);
    end
  end
  
  if ~isempty(rawdata),
    fnsnew = fieldnames(rawdatacurr);
    fnsold = fieldnames(rawdata);
    fnsnew1 = setdiff(fnsnew,fnsold);
    if ~isempty(fnsnew1),
      fprintf('New fields in file %s that are ignored:\n',ppfiles{i});
      fprintf('%s\n',fnsnew1{:});
      rawdatacurr = rmfield(rawdatacurr,fnsnew1);
    end
    fnsold1 = setdiff(fnsold,fnsnew);
    if ~isempty(fnsold1),
      fprintf('Missing fields in file %s that will be removed from previous files:\n',ppfiles{i});
      fprintf('%s\n',fnsold1{:});
      rawdata = rmfield(rawdata,fnsold1);
    end    
  end
  
  assert(all(isfield(rawdatacurr,behaviorfns)));
  assert(all(isfield(rawdatacurr,requiredfns)));
  rawdata = structappend(rawdata,rawdatacurr(:));

end

rawdata0 = rawdata;

ismissing = ~ismember({rawdata.expdir},{trxdata.expdir});
if any(ismissing)
  fprintf('%d experiments (%d laser, %d tone) in post-processed data and not in trxdata\n',...
    nnz(ismissing),nnz(~strcmp({rawdata(ismissing).trialtype},'CTR')),nnz(strcmp({rawdata(ismissing).trialtype},'CTR')));
  
  for mousei = 1:numel(mice),
    mouseidxcurr = find(strcmp({rawdata.mouse},mice{mousei}));
    [dayscurr,~,dayidxcurr] = unique({rawdata(mouseidxcurr).day});
    for dayi = 1:numel(dayscurr),
      idxcurr = mouseidxcurr(dayidxcurr==dayi);
      ismissingcurr = ismissing(idxcurr);
      if any(ismissingcurr),
        fprintf('%d/%d experiments from %s %s (%d/%d laser, %d/%d tone) in post-processed data and not in trxdata\n',...
          nnz(ismissingcurr),numel(ismissingcurr),mice{mousei},dayscurr{dayi},...
          nnz(~strcmp({rawdata(idxcurr(ismissingcurr)).trialtype},'CTR')),...
          nnz(~strcmp({rawdata(idxcurr).trialtype},'CTR')),...
          nnz(strcmp({rawdata(idxcurr(ismissingcurr)).trialtype},'CTR')),...
          nnz(strcmp({rawdata(idxcurr).trialtype},'CTR')));
        %fprintf('%s\n',rawdata(idxcurr(ismissingcurr)).expdir);

      end
    end
  end
end
ismissing = ~ismember({trxdata.expdir},{rawdata.expdir});

for i = 1:numel(trxdata),
  [~,n] = fileparts(trxdata(i).expdir);
  m = regexp(n,'^([^_]*)_(\d{8})','once','tokens');
  trxdata(i).mouse = m{1};
  trxdata(i).day = m{2};
end

if any(ismissing)
  fprintf('%d experiments (%d laser, %d tone) in trx data and not in post-processed data\n',...
    nnz(ismissing),nnz([trxdata(ismissing).islaser]),nnz([trxdata(ismissing).istone]));
  
  for mousei = 1:numel(mice),
    mouseidxcurr = find(strcmp({trxdata.mouse},mice{mousei}));
    [dayscurr,~,dayidxcurr] = unique({trxdata(mouseidxcurr).day});
    for dayi = 1:numel(dayscurr),
      idxcurr = mouseidxcurr(dayidxcurr==dayi);
      ismissingcurr = ismissing(idxcurr);
      if any(ismissingcurr),
        fprintf('%d/%d experiments from %s %s (%d/%d laser, %d/%d tone) in trxdata and not in post-processed data\n',...
          nnz(ismissingcurr),numel(ismissingcurr),mice{mousei},dayscurr{dayi},...
          nnz([trxdata(idxcurr(ismissingcurr)).islaser]),...
          nnz([trxdata(idxcurr).islaser]),...
          nnz([trxdata(idxcurr(ismissingcurr)).istone]),...
          nnz([trxdata(idxcurr).istone]));
        %fprintf('%s\n',trxdata(idxcurr(ismissingcurr)).expdir);
      end
    end
  end
  
end

if false,
  assert(isempty(setxor({rawdata.expdir},{trxdata.expdir})));
end


%% line up 

rawdata = rawdata0;
trxdata = trxdata0;

% line these up
[ism,idx] = ismember({trxdata.expdir},{rawdata.expdir}); % AL20141205: warning, changed definition of .id produced by ExpPP due to collisions
if ~all(ism),
  fprintf('The following exps for trajectory info do not have corresponding ids for the behavior data (%d trials):\n',nnz(~ism));
  fprintf('%s\n',trxdata(~ism).expdir);
end
ism2 = ismember({rawdata.expdir},{trxdata.expdir}); 
if ~all(ism2),
  fprintf('The following ids for the behavior data not have corresponding ids for trajectory info (%d trials):\n',nnz(~ism2));
  fprintf('%s\n',rawdata(~ism2).expdir);
end
rawdata = rawdata(idx(ism));
trxdata = trxdata(ism);

assert(all(strcmp({rawdata.expdir},{trxdata.expdir})));

for i = 1:numel(trxdata),
  
  trxdata(i).mouse = rawdata(i).mouse;
  
end

%% make day either tone or laser

for i = 1:numel(rawdata),
  
  rawdata(i).real_day = rawdata(i).day;
  if trxdata(i).istone,
    assert(~trxdata(i).islaser);
    rawdata(i).day = 'tone';
  else
    assert(trxdata(i).islaser);
    rawdata(i).day = 'laser';
  end
  rawdata(i).session_day = rawdata(i).day;
  trxdata(i).session_day = rawdata(i).day;
end

%% load in timing info
% 
% for i = 1:numel(rawdata),
%   rawdata(i).laseron = nan;
%   rawdata(i).laseroff = nan;
% end
% 
% for i = 1:numel(laserinfofiles),
%   fid = fopen(laserinfofiles{i},'r');
%   while true,
%     s = fgetl(fid);
%     if ~ischar(s),
%       break;
%     end
%     ss = regexp(s,',','split');
%     if numel(ss) < laseroffcolumn(i),
%       fprintf('Not enough columns in >>%s<<, skipping\n',s);
%       continue;
%     end
%     exp = ss{laserexpnamecolumn(i)};
%     if isempty(exp),
%       continue;
%     end
%     m = regexp(exp,'M\d+_\d{8}_v\d{3}','once');
%     if isempty(m),
%       fprintf('Bad exp name >>%s<<, skipping\n',exp);
%       continue;
%     end
%     laseroncurr = str2double(ss{laseroncolumn(i)});
%     if isnan(laseroncurr),
%       fprintf('Bad laser on time >>%s<<, skipping\n',ss{laseroncolumn(i)});
%       continue;
%     end
%     laseroffcurr = str2double(ss{laseroffcolumn(i)});
%     if isnan(laseroffcurr),
%       fprintf('Bad laser off time >>%s<<, skipping\n',ss{laseroffcolumn(i)});
%       continue;
%     end
%     j = find(strcmp({rawdata.exp},exp));
%     if isempty(j),
%       fprintf('Exp %s not in rawdata, skipping\n',exp);
%       continue;
%     end
%     assert(numel(j)==1);
%     fprintf('SUCCESS! %s adding laseron = %d, laseroff = %d\n',exp,laseroncurr,laseroffcurr);
%     rawdata(j).laseron = laseroncurr;
%     rawdata(j).laseroff = laseroffcurr;
%   end
%   fclose(fid);
% end
% 
% idx = find(strcmp({rawdata.date},'CueLaser4sec') & isnan([rawdata.laseron]));
% if ~isempty(idx),
%   fprintf('For %d experiments within CueLaser4sec subdirs, setting laseron = 0, laseroff = 2000\n',numel(idx));
%   fprintf('%s\n',rawdata(idx).exp);
%   for i = idx(:)',
%     rawdata(i).laseron = 0;
%     rawdata(i).laseroff = 2000;
%   end
% end
% 
% idx = find(strcmp({rawdata.date},'CueLaser2sec') & isnan([rawdata.laseron]));
% if ~isempty(idx),
%   fprintf('For %d experiments within CueLaser2sec subdirs, setting laseron = 0, laseroff = 2000\n',numel(idx));
%   fprintf('%s\n',rawdata(idx).exp);
%   for i = idx(:)',
%     rawdata(i).laseron = 0;
%     rawdata(i).laseroff = 1000;
%   end
% end
% 
% [mice,~,mouseidx] = unique({rawdata.mouse});
% for mousei = 1:numel(mice),
%   ismissing = isnan([rawdata.laseron]) & (mouseidx==mousei) & [trxdata.islaser];
%   fprintf('%s: %d exps missing laser timing info.\n',mice{mousei},nnz(ismissing));
%   fprintf('%s\n',rawdata(ismissing).exp);
% end

%% look for side trajectories on the right or front trajectories on the left

[readframe,~,fid] = get_readframe_fcn(fullfile(trxdata(1).expdir,moviefilestr));
im = readframe(1);
imsz = size(im);
if fid > 1,
  fclose(fid);
end

expsremove = false(1,numel(trxdata));
for expi = 1:numel(trxdata),
  idxbad = find(trxdata(expi).x1>=imsz(2)/2);
  if ~isempty(idxbad),
    fprintf('Bad side view trajectory for %s, frames %s\n',rawdata(expi).exp,mat2str(idxbad));
    expsremove(expi) = true;
  end
  idxbad = find(trxdata(expi).x2<=imsz(2)/2);
  if ~isempty(idxbad),
    fprintf('Bad front view trajectory for %s, frames %s\n',rawdata(expi).exp,mat2str(idxbad));
    expsremove(expi) = true;
  end
end

trxdata(expsremove) = [];
rawdata(expsremove) = [];

% Bad side view trajectory for M134_20150303_v003, frames [137;138]
% Bad front view trajectory for M134_20150303_v003, frames [137;138]


[mice,~,mouseidx] = unique({rawdata.mouse});
for mousei = 1:numel(mice),
  fprintf('Mouse %s, %d tone, %d laser experiments\n',mice{mousei},nnz([trxdata(mouseidx==mousei).istone]),nnz([trxdata(mouseidx==mousei).islaser]));
end

%% remove videos where the mouse's paw is not on the perch at the start of lift 0 

nexps = numel(rawdata);

maxdist_median_liftpos = 30;

x1 = nan(1,nexps);
y1 = nan(1,nexps);

behaviori = find(strcmp(behaviors,'Lift'));
behaviorfn = behaviorfns{behaviori};

for i = 1:nexps,
  
  if isnan(rawdata(i).(behaviorfn)),
    continue;
  end
  
  x1(i) = trxdata(i).x1(rawdata(i).(behaviorfn));
  y1(i) = trxdata(i).y1(rawdata(i).(behaviorfn));
  
end

[mice,~,mouseidx] = unique({rawdata.mouse});
mux = nan(1,numel(mice));
muy = nan(1,numel(mice));
d = nan(1,nexps);
for mousei = 1:numel(mice),
  mux(mousei) = nanmedian(x1(mouseidx==mousei));
  muy(mousei) = nanmedian(y1(mouseidx==mousei));
  d(mouseidx==mousei) = sqrt((x1(mouseidx==mousei)-mux(mousei)).^2 + (y1(mouseidx==mousei)-muy(mousei)).^2);
end

badidx = d > maxdist_median_liftpos;

fprintf('Removing the following experiments:\n');
d(isnan(d)) = 0;
[~,order] = sort(d,'descend');
for i = 1:nnz(badidx),
  j = order(i);
  fprintf('%s, frame %d, d = %f\n',rawdata(j).expdir,rawdata(j).(behaviorfn),d(j));
end

rawdata(badidx) = [];
trxdata(badidx) = [];


nexps = numel(rawdata);


[mice,~,mouseidx] = unique({rawdata.mouse});
for mousei = 1:numel(mice),
  fprintf('Mouse %s, %d tone, %d laser experiments\n',mice{mousei},nnz([trxdata(mouseidx==mousei).istone]),nnz([trxdata(mouseidx==mousei).islaser]));
end

%% video frame from each day

[mice,~,mouseidx] = unique({rawdata.mouse});

% tmp = [rawdata.auto_GS00_Lift_0];
% tmp = tmp(~isnan(tmp));
% imt = round(prctile(tmp,10));

ims = cell(1,numel(mice));
imdays = cell(1,numel(mice));
for mousei = 1:numel(mice),
  
  mouseidxcurr = find(mouseidx==mousei);
  [imdays{mousei},~,dayidx] = unique({rawdata(mouseidxcurr).day});
  idxcurr = mouseidxcurr(dayidx==1);
  xgrab = nan(4,numel(idxcurr));
  for ii = 1:numel(idxcurr),
    i = idxcurr(ii);
    if isnan(rawdata(i).auto_GS00_Grab_0),
      continue;
    end
    xgrab(:,ii) = [trxdata(i).x1(rawdata(i).auto_GS00_Grab_0)
      trxdata(i).y1(rawdata(i).auto_GS00_Grab_0)
      trxdata(i).x2(rawdata(i).auto_GS00_Grab_0)
      trxdata(i).y2(rawdata(i).auto_GS00_Grab_0)];
  end
  mu = nanmedian(xgrab,2);
  [~,j] = min(sum(bsxfun(@minus,xgrab,mu).^2,1),[],2);
  j = idxcurr(j);
%   if strcmp(mice{mousei},'M134'),
%     j = idxcurr(2);
%     t = rawdata(j).auto_GS00_Grab_0;
%   else
%    j = idxcurr(find(~isnan([rawdata(idxcurr).auto_GS00_Grab_0]),1));
%   end
  t = rawdata(j).auto_GS00_Grab_0;
  expdir = rawdata(j).expdir;
  readframe = get_readframe_fcn(fullfile(expdir,moviefilestr));
  fprintf('Mouse %s, %s\n',mice{mousei},rawdata(j).exp);
  im = readframe(t);
  ims{mousei} = repmat(im,[1,1,1,numel(imdays{mousei})]);
end

%% 3D reconstruction

[~,mouseidx] = ismember({rawdata.mouse},calibrationdata.mice);
for expi = 1:numel(rawdata),
  mousei = mouseidx(expi);
  xL = [trxdata(expi).x1';trxdata(expi).y1'];
  xR = [trxdata(expi).x2'-imsz(2)/2;trxdata(expi).y2'];
  if mousei == 0,
    omcurr = calibrationdata.om0;
    Tcurr = calibrationdata.T0;
  else
    omcurr = calibrationdata.ompermouse(:,mousei);
    Tcurr = calibrationdata.Tpermouse(:,mousei);
  end
  XL = stereo_triangulation(xL,xR,omcurr,Tcurr,calibrationdata.fc_left,...
    calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left,...
    calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,...
    calibrationdata.alpha_c_right);
  XL = bsxfun(@minus,XL,calibrationdata.origin(:,mousei));
  trxdata(expi).X = XL(1,:);
  trxdata(expi).Y = XL(2,:);
  trxdata(expi).Z = XL(3,:);
end

%% compute various statistics as a function of path length


fracsoffplot = linspace(0,1,4);
allfracsoff = linspace(0,1,100);

[trxdata,rawdata] = AlignTrajectoriesByPathLength(trxdata,rawdata,behaviors,behaviorfns,'Lift','Grab',allfracsoff);
%[trxdata,rawdata] = AlignTrajectoriesByPathLength3D(trxdata,rawdata,behaviors,behaviorfns,'Lift','Grab',allfracsoff);


%% histogram number of trials per date

hfig = 2;
figure(hfig);
clf;
hax = createsubplots(numel(mice),1,.05);

for mousei = 1:numel(mice),
  
  mouseidxcurr = find(strcmp({rawdata.mouse},mice{mousei}));
  isctr = strcmp({rawdata(mouseidxcurr).trialtype},'CTR');
  [dayscurr,~,dayidxcurr] = unique({rawdata(mouseidxcurr).real_day});
  counts_ctr = hist(dayidxcurr(isctr),1:numel(dayscurr));
  counts_laser = hist(dayidxcurr(~isctr),1:numel(dayscurr));
  
  axes(hax(mousei));
  bar([counts_ctr;counts_laser]');
  set(hax(mousei),'XTick',1:numel(dayscurr),'XTickLabel',dayscurr,'Box','off');  
  title(mice{mousei});
  
end

axes(hax(1));
legend('Control','Laser');

save data/ToneVsLaserData20150717.mat rawdata trxdata;

rawdata1 = rawdata;
trxdata1 = trxdata;


%% date match all data: only choose control trials from days where we have laser trials

dokeep = false(1,numel(rawdata1));
for mousei = 1:numel(mice),
  
  mouseidxcurr = find(strcmp({rawdata1.mouse},mice{mousei}));
  isctr = strcmp({rawdata1(mouseidxcurr).trialtype},'CTR');
  [dayscurr,~,dayidxcurr] = unique({rawdata1(mouseidxcurr(~isctr)).real_day});
  [ism] = ismember({rawdata1(mouseidxcurr(isctr)).real_day},dayscurr);
  dokeep(mouseidxcurr(isctr)) = ism;
  dokeep(mouseidxcurr(~isctr)) = true;

end

rawdata = rawdata1(dokeep);
trxdata = trxdata1(dokeep);

%% plot a bunch of control success 3d trajectories

nplot = 50;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors,behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);

for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulGrabs3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
  
end


nplot = 30;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors,behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);
for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulGrabs3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
  
end


nplot = 10;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors,behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);
for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulGrabs3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
  
end

%% plot a bunch of control success grab trajectories

nplot = 50;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors(1:3),behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);
for mousei = 1:numel(mice),
  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulLift2Grab3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
end


nplot = 30;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors(1:3),behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);
for mousei = 1:numel(mice),
  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulLift2Grab3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
end


nplot = 10;
[hfigs,mice] = PlotSuccessfulTrajectories3D(trxdata,rawdata,behaviors(1:3),behaviorfns,imsz,'iscnofns',{'islaser'},'daysplot',{'tone'},'nplot',nplot);
set(hfigs,'Units','pixels','Position',[50 1067 591 419]);
for mousei = 1:numel(mice),
  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllSuccessfulLift2Grab3D_%dtrx_%s',nplot,mice{mousei})),{'pdf','png'});
end

%% plot grab trajectories

[hfigs,mice] = PlotFirstGrabTrajectories(trxdata,rawdata,behaviors,behaviorfns,imsz,'figbase',250,'ims',ims,'imdays',imdays,...
  'iscnofns',{'islaser'}, 'controltypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'cnotypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},'nframespost',0,'nframespre',0,...
  'matchsamplesize',true);

for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  figure(hfig);
  truesize;
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllGrabs_%s',mice{mousei})),{'pdf','png'});
  
end

[hfigs,mice] = PlotGrab2AtmouthTrajectories(trxdata,rawdata,behaviors,behaviorfns1,imsz,'figbase',300,'ims',ims,'imdays',imdays,...
  'iscnofns',{'islaser'}, 'controltypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'cnotypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},'nframespost',0,'nframespre',0,...
   'matchsamplesize',true);



for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  figure(hfig);
  truesize;
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('AllGrab2Atmouths_%s',mice{mousei})),{'pdf','png'});
  
end

%% plot position distribution difference over time

chi2fracsoff = linspace(0,1,11);
[nnaccuracy,balancednnaccuracy,nnchanceaccuracy,nnpval,mice,hfigs] = ...
  PlotPositionDistributionDistance(trxdata,rawdata,behaviors,behaviorfns,'Lift','Grab',...
  'fracsoff',chi2fracsoff,...
  'iscnofns',{'islaser'}, 'controltypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'cnotypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'detectoutliers',false); 

for mousei = 1:numel(mice),

  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('NearestNeighborTrajClassificationVsChance%s',mice{mousei})),{'pdf','png'});
  
end

%% plot end points

chi2fracsoff = linspace(0,1,101);
[nnaccuracy,balancednnaccuracy,nnchanceaccuracy,nnpval,mice,hfigs] = ...
  PlotPositionDistributionDistance(trxdata,rawdata,behaviors,behaviorfns,'Lift','Grab',...
  'fracsoff',chi2fracsoff,...
  'iscnofns',{'islaser'}, 'controltypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'cnotypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'detectoutliers',false); 

nnmutinf = max(0,bsxfun(@minus,...
  reshape(-nnchanceaccuracy.*log2(nnchanceaccuracy)-(1-nnchanceaccuracy).*log2(1-nnchanceaccuracy),[1,1,numel(mice)]),...
  -nnaccuracy.*log2(nnaccuracy)-(1-nnaccuracy).*log2(1-nnaccuracy)));

[hfigs,mice,axlims] = PlotPosition2(trxdata,rawdata,imsz,behaviors,behaviorfns,'Grab',fracsoffplot,'hfigbase',3160,'ims',ims,'imdays',imdays,'plotstderr',false,...
  'offsetby','fracpath','behavior0','Lift','colorscheme',2,'markersize',7,'markersizemean',12,...
  'iscnofns',{'islaser'}, 'controltypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'cnotypes',{'successful_1grab','successful_2plusgrab','unsuccessful'},...
  'legendstrs',{'Laser','Tone'},...
  'chi2dist',balancednnaccuracy,'chi2fracsoff',chi2fracsoff,'testtype','Balanced NN Accuracy',...
  'axlims',.1);

for mousei = 1:numel(mice),
  hfig = hfigs(mousei);
  SaveFigLotsOfWays(hfig,fullfile(savefigdir,sprintf('PositionAlongLift2Grab_%s',mice{mousei})),{'pdf','png'});
end

%% compute median positions for each behavior

meanpos = struct;
[mice,~,mouseidx] = unique({rawdata.mouse});
for mousei = 1:numel(mice),
  idxcurr = find(mouseidx==mousei & [trxdata.istone] & strcmp({rawdata.auto_Grab_successtype},'successful_1grab'));
  fprintf('Computing mean positions for mouse %s from %d trials\n',mice{mousei},numel(idxcurr));
  for behi = 1:nbehaviors,
    behavior = behaviors{behi};
    x = nan(7,numel(idxcurr));
    for expii = 1:numel(idxcurr),
      expi = idxcurr(expii);
      t = rawdata(expi).(behaviorfns{behi});
      if ~isnan(t),
        x(:,expii) = [trxdata(expi).x1(t);trxdata(expi).y1(t);...
          trxdata(expi).x2(t);trxdata(expi).y2(t);...
          trxdata(expi).X(t);trxdata(expi).Y(t);trxdata(expi).Z(t)];
      end
    end
    mu = nanmedian(x,2);
    meanpos(mousei).(behavior).x1 = mu(1);
    meanpos(mousei).(behavior).y1 = mu(2);
    meanpos(mousei).(behavior).x2 = mu(3);
    meanpos(mousei).(behavior).y2 = mu(4);
    meanpos(mousei).(behavior).X = mu(5);
    meanpos(mousei).(behavior).Y = mu(6);
    meanpos(mousei).(behavior).Z = mu(7);
  end
  
end

save('data/MeanCTRBehaviorPosition20150717.mat','meanpos','mice');

%% compute minimum distance to perch within x frames after chew

dts = 125:125:1000;
mindperch = nan(numel(rawdata),numel(dts));

for expi = 1:numel(rawdata),
  
  t0 = rawdata(expi).auto_GSSS_Chew_0;
  if isnan(t0),
    continue;
  end
  mousei = find(strcmp(rawdata(expi).mouse,mice));
  
  for dti = 1:numel(dts),
    
    t1 = t0 + dts(dti) - 1;
    if t1 > numel(trxdata(expi).X),
      continue;
    end
    mindperch(expi,dti) = min(sqrt( (trxdata(expi).X(t0:t1) - meanpos(mousei).Lift.X).^2 + ...
      (trxdata(expi).Y(t0:t1) - meanpos(mousei).Lift.Y).^2 + ...
      (trxdata(expi).Z(t0:t1) - meanpos(mousei).Lift.Z).^2 ));
    
  end
  
end

fid = fopen(fullfile(savefigdir,'MinDist2Perch_ToneVsLaser.csv'),'w');
fprintf(fid,'Experiment name');
for i = 1:numel(dts),
  fprintf(fid,',dt=%d',dts(i));
end
fprintf(fid,'\n');
for expi = 1:numel(rawdata),
  
  fprintf(fid,'%s',rawdata(expi).exp);
  for i = 1:numel(dts),
    fprintf(fid,',%f',mindperch(expi,i));
  end
  fprintf(fid,'\n');
end
fclose(fid);

%% plot results for one video

expdir = '/tier2/hantman/Jay/videos/M134C3VGATXChR2/20141216L/CTR/M134_20141216_v011';
MakeMouseTrackingResultsVideo(expdir,trxdata,'endframe',500,'resvideo','M134_20141216_v011_trx.avi','fps',50)