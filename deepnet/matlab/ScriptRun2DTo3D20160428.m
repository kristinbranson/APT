%% paths

% addpath /groups/branson/bransonlab/mayank/JAABA/filehandling;
% addpath /groups/branson/bransonlab/mayank/JAABA/misc;
addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;

addpath /groups/branson/bransonlab/projects/flyHeadTracking/code/
addpath ~bransonk/tracking/code/Ctrax/matlab/netlab

%% 

% dd = dir('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/*_side.avi');
dd = dir('/nobackup/branson/mayank/stephenOut/*_side.mat');
expnames = {};

for ndx = 35:numel(dd)
  ii = dd(ndx).name(1:end-9);
  [xx,~,~,fstr] = regexp(dd(ndx).name,'fly(_*\d+)');
  assert(~isempty(xx),'filename is weird');
  fparts = strsplit(dd(ndx).name(1:end-9),'__');
  if numel(fparts)>2,
    fparts{3} = fparts{3}(2:end);
  end
  experiment_name = ii;
  if strcmp(ii(1:10),'PoseEstima'),
    bdir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/';
    
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C002H001S' fparts{3}],['C002H001S' fparts{3} '.avi']);
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C001H001S' fparts{3}],['C001H001S' fparts{3} '.avi']);
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kdd = dir(fullfile(bdir,fstr{1},'kineData','*.mat'));
    assert(numel(kdd)==1,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
    kinematfile = fullfile(bdir,fstr{1},'kineData',kdd(1).name);
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
  elseif strcmp(ii(1:10),'flyHeadTra')
    ff = fopen('/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/FlyNumber2CorrespondingDLTfile.csv','r');
    K = textscan(ff,'%d,%s');
    fclose(ff);
    flynum = str2double(fstr{1}(4:end));
    mndx = find(K{1}==flynum);
    if isempty(mndx),
      fprintf('%d:%s dont have kinedata .. skipping\n',ndx,dd(ndx).name);
      continue;
    end
    bdir = '/groups/branson/bransonlab/projects/flyHeadTracking/ExamplefliesWithNoTrainingData/';
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C002H001S' fparts{3}],['C002H001S' fparts{3} '_c.avi']);
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C001H001S' fparts{3}],['C001H001S' fparts{3} '_c.avi']);
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kinematfile = fullfile('/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/DLTs/',[K{2}{mndx} '_kine.mat']);
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
  
  elseif strcmp(ii(1:10),'projects__'),
    bdir = '/groups/branson/bransonlab/projects/flyHeadTracking/';
    kdd = dir(fullfile(bdir,fstr{1},'kineData','*_kineData.mat'));
    if numel(kdd)==0,
      kdd = dir(fullfile(bdir,fstr{1},'kineData_300ms_stimuli','*kineData*.mat'));
      assert(numel(kdd)==1,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
      kinematfile = fullfile(bdir,fstr{1},'kineData_300ms_stimuli',kdd(1).name);
    else
      kdd = kdd(1);
%       assert(numel(kdd)==1,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
      kinematfile = fullfile(bdir,fstr{1},'kineData',kdd(1).name);
    end
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C002H001S' fparts{3}],['C002H001S' fparts{3} '_c.avi']);
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C001H001S' fparts{3}],['C001H001S' fparts{3} '_c.avi']);
    frontviewmatfile = fullfile('/nobackup/branson/mayank/stephenOut/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/nobackup/branson/mayank/stephenOut/',[dd(ndx).name(1:end-4) '.mat']);
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
    
  else
    bdir = ['/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/' fparts{1} '/data/'];
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C002H001S' fparts{3}],['C002H001S' fparts{3} '_c.avi']);
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),['C001H001S' fparts{3}],['C001H001S' fparts{3} '_c.avi']);
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kdd = fullfile(bdir,'kineData',['kinedata_' fparts{2}(2:end) '.mat']);
    assert(exist(kdd,'file')>0,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
    kinematfile = kdd;
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
  end
  
  if ~exist(frontviewvideofile,'file'),
    fprintf('Didnt find front video file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(sideviewvideofile,'file'),
    fprintf('Didnt find side view video file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(frontviewmatfile,'file'),
    fprintf('Didnt find front view mat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(sideviewmatfile,'file'),
    fprintf('Didnt find side view mat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(kinematfile,'file'),
    fprintf('Didnt find kinemat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
%   if ~exist(frontviewresultsvideofile,'file'),
%     fprintf('Didnt find frontviewresultsvideofile file for %d %s\n',ndx,dd(ndx).name);
%     continue;
%   end
  
  [fdir,~] = fileparts(frontviewvideofile);
  savefile = [fdir,'_3Dres.mat'];
  if exist(savefile,'file') && ~redo,
    continue;
  end
  
  compute3Dfrom2D(savefile,frontviewvideofile,sideviewvideofile,frontviewmatfile,sideviewmatfile,kinematfile,false,experiment_name);
  
%   Script2DTo3DTracking20160409;
end
