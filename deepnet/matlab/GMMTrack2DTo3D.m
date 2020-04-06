function GMMTrack2DTo3D(fmoviefile,smoviefile,kinelistfile,outdir,redo)

%% Constructs smoothed 3D construction for videos listed in file

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;

addpath /groups/branson/bransonlab/projects/flyHeadTracking/code/
addpath /groups/branson/home/bransonk/tracking/code/Ctrax/matlab/netlab

% infile = '~/bransonlab/PoseEstimationData/Stephen/folders2track.txt';
% outdir = '/nobackup/branson/mayank/stephenOut/';
% kinelistfile = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/FlyNumber2CorrespondingDLTfile.csv';
% kineDir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/DLTs/';

if nargin < 5,
  redo = false;
end
%% read the list of files
fidf = fopen(fmoviefile,'r');
fids = fopen(smoviefile,'r');

Xf = textscan(fidf,'%s');
Xf = Xf{1};
Xs = textscan(fids,'%s');
Xs = Xs{1};
fclose(fidf);fclose(fids);

assert(numel(Xf)==numel(Xs),'Number of movies in front view file and side view text file should be same');

ff = fopen(kinelistfile,'r');
K = textscan(ff,'%d,%s');
fclose(ff);

%%

parfor ndx = 1:numel(Xf)
  fparts = strsplit(Xf{ndx},filesep);
  outf = sprintf('%s__%s__%s_front.mat',fparts{end-5},fparts{end-2},fparts{end-1}(end-3:end));
  fparts = strsplit(Xs{ndx},filesep);
  outs = sprintf('%s__%s__%s_side.mat',fparts{end-5},fparts{end-2},fparts{end-1}(end-3:end));
  matfilef = fullfile(outdir,outf);
  matfiles = fullfile(outdir,outs);
  if ~exist(matfilef,'file') || ~exist(matfiles,'file'),
    fprintf('Outfile %s or %s dont exist for %s\n',matfilef,matfiles,Xf{ndx})
    continue;
  end
  flynum = str2double(fparts{end-2}(4:6));
  mndx = find(K{1}==flynum);
  if isempty(mndx),
    fprintf('%d:%s dont have kinedata .. skipping\n',ndx,Xf{ndx});
    continue;
  end
  
  kinematfile = K{2}{mndx};
  
  outfile = [matfiles(1:end-9) '_3Dres.mat'];
  
  if ndx<0,
    makevideo = true;
  else
    makevideo = false;
  end

  if ~(exist(outfile,'file') && ~redo),
    
    [~,dname] = fileparts(Xf{ndx});
    fmov = fullfile(Xf{ndx},[dname '_c.avi']);
    [~,dname] = fileparts(Xs{ndx});
    smov = fullfile(Xs{ndx},[dname '_c.avi']);
    fprintf('Working on %d:%s\n',ndx,Xf{ndx});
    compute3Dfrom2D(outfile,fmov,smov,matfilef,matfiles,kinematfile,makevideo);
    fprintf('Done.\n');
  end

  % convert back to 2D
  matfile = outfile;
  if ~exist(matfile,'file'),
    fprintf('3D mat file %s dont exist for %s\n',matfile,Xf{ndx})
    continue;
  end
  ftrx = [Xf{ndx}(1:end-4) '.trk']
  strx = [Xs{ndx}(1:end-4) '.trk']
  
  convertResultsToTrx(matfile,ftrx,strx);
  fprintf('Done conversion for %d %s\n',ndx,Xf{ndx});
  
end
