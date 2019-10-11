% set up paths
addpath ..;
addpath ../misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc/
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling/

rootdatadir = '/groups/branson/bransonlab/projects/flyHeadTracking';
expnames = {
  '5_6_13_81B12AD_47D05DBD_x_Chrimsonattp18'
  '7_7_13_47D05AD_81B12DBD_x_Chrimsonattp18'
  'SallyDigitizedVideos'
  };

matfiles = {};
vid1files = {};
vid2files = {};
expidx = [];

framewidthrange = [1024,1024];
frameheightrange = [1024,1024];

for expi = 1:numel(expnames),
  expdir = fullfile(rootdatadir,expnames{expi},'data');
  flydirs = mydir(expdir,'isdir',true,'name','^fly_\d+$');
  kinedir = fullfile(expdir,'kineData');
  iskinedir = exist(kinedir,'dir');
  if iskinedir,
    kinematfiles = mydir(kinedir,'isdir',false,'name','.*[Dd]ata.*fly.*trial.*\.mat$');
  end
    
  for flyi = 1:numel(flydirs),
    
    flydir = flydirs{flyi};
    trialdirs = mydir(flydir,'isdir',true,'name','_trial_\d+$');
    
    for triali = 1:numel(trialdirs),
      trialdir = trialdirs{triali};
      
      [~,n] = fileparts(trialdir);
      m = regexp(n,'^fly_(\d+)_trial_(\d+)$','once','tokens');
      assert(~isempty(m));
      fly = str2double(m{1});
      trial = str2double(m{2});
      
      viddir1 = mydir(trialdir,'isdir',true,'name','^C001');
      assert(numel(viddir1)==1);
      viddir1 = viddir1{1};
      vidfile1 = mydir(viddir1,'name','^C001.*\.avi');
      assert(numel(vidfile1)==1);
      vidfile1 = vidfile1{1};
      vidinfo = aviinfo(vidfile1); %#ok<FREMO>
      if vidinfo.Width < framewidthrange(1) || vidinfo.Width > framewidthrange(2) || ...
          vidinfo.Height < frameheightrange(1) || vidinfo.Height > frameheightrange(2),
        fprintf('Wrong frame size for %s, skipping\n',vidfile1);
        continue;
      end
      
      viddir2 = mydir(trialdir,'isdir',true,'name','^C002');
      assert(numel(viddir2)==1);
      viddir2 = viddir2{1};
      vidfile2 = mydir(viddir2,'name','^C002.*\.avi');
      assert(numel(vidfile2)==1);
      vidfile2 = vidfile2{1};
      vidinfo = aviinfo(vidfile2); %#ok<FREMO>
      if vidinfo.Width < framewidthrange(1) || vidinfo.Width > framewidthrange(2) || ...
          vidinfo.Height < frameheightrange(1) || vidinfo.Height > frameheightrange(2),
        fprintf('Wrong frame size for %s, skipping\n',vidfile2);
        continue;
      end
      
      if iskinedir,

        i = find(~cellfun(@isempty,regexp(kinematfiles,sprintf('fly_0*%d_trial_0*%d\\.mat',fly,trial),'once')));
        assert(numel(i)<=1);
        if ~isempty(i),
          kinematfile = kinematfiles{i};
        else
          fprintf('No kine mat file found for trial dir %s, skipping\n',trialdir);
        end
        
      else
        
        kinematfile = mydir(trialdir,'isdir',false,'name','kinedata.mat$');
        if isempty(kinematfile),
          kinematfile = mydir(trialdir,'isdir',false,'name','data.mat$');
        end
        if isempty(kinematfile),
          fprintf('No kine mat file found in %s, skipping\n',trialdir);
          continue;
        end
        assert(numel(kinematfile)==1);
        kinematfile = kinematfile{1};
        
      end
      
      matfiles{end+1} = kinematfile;
      vid1files{end+1} = vidfile1;
      vid2files{end+1} = vidfile2;
      expidx(end+1) = expi;
      
    end
    
  end
  
end

nperexp = hist(expidx,1:numel(expnames));
for expi = 1:numel(expnames),
  fprintf('%s: %d experiments found\n',expnames{expi},nperexp(expi));
end

savedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
savefile = fullfile(savedir,'FlyHeadStephenTestData_20150813.mat');

%% put all the data in one mat file

labels = struct;
labels.pts = [];
labels.ts = [];
labels.expidx = [];
labels.vid1files = vid1files;
labels.vid2files = vid2files;
labels.matfiles = matfiles;

for i = 1:numel(matfiles),

  res = get2DpointsFromKine('headDataMatFile',matfiles{i},'debug',0);
  if ~isfield(res,'headView1'),
    continue;
  end
  if isempty(labels.pts),
    npts = size(res.headView1,1);
    labels.pts = nan([2,2,npts,0]);
  end
  % (x,y) x view x npts x n
  n = numel(res.frameidx);
  labels.pts(:,:,:,end+1:end+n) = cat(2,permute(res.headView1,[2,4,1,3]),permute(res.headView2,[2,4,1,3]));
  labels.ts(end+1:end+n) = res.frameidx;
  labels.expidx(end+1:end+n) = i;
  
end

%% save

save(savefile,'-struct','labels');
