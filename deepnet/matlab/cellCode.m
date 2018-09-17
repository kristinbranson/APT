%% Compute 3d tracking

addpath netlab
expdir = '/home/mayank/work/PoseEstimationData/Stephen/fly219/fly219_trial1/';
savefile = fullfile(expdir,'Out_3d.mat');
fvidfile = fullfile(expdir,'C002H001S0001','C002H001S0001_c.avi');
svidfile = fullfile(expdir,'C001H001S0001','C001H001S0001_c.avi');
fmat = fullfile(expdir,'C002H001S0001','projects__fly219_trial1__0001.mat');
smat = fullfile(expdir,'C001H001S0001','projects__fly219_trial1__0001_side.mat');
kinefile = fullfile(expdir,'01_fly219_kineData_kine.mat');
dosave = false;

compute3Dfrom2D(savefile,fvidfile,svidfile,fmat,smat,kinefile,dosave);

%% 3d back to 2d

ftrk = fullfile(expdir,'C002H001S0001','C002H001S0001_c.trk');
strk = fullfile(expdir,'C001H001S0001','C001H001S0001_c.trk');

convertResultsToTrx(savefile,ftrk,strk);

%% for body axis

if ~exist('get_readframe_fcn')
  addpath ~/work/JAABA/filehandling;
  addpath ~/work/JAABA/misc;
end
odir = '../data/out/';
dds = dir(fullfile(odir,'*_side.mat'));
L = load('../data/bodyTracking/labels.mat');
f = figure(1);
cc = jet(5);
scale = 8;
for ndx = 1:numel(dds)
  Q = load(fullfile(odir,[dds(ndx).name(1:end-8) 'front.mat']));
  mndx = [];
  for ii = 1:numel(L.vid1files)
    kk = strsplit(L.vid1files{ii},'/');
    kk(cellfun(@isempty,kk))=[];
    oname = [kk{end-5} '__' kk{end-2} '__' kk{end-1}(end-3:end) '_side.mat'];
    if strcmp(oname,dds(ndx).name),
      mndx = ii; break;
    end
  end
  assert(~isempty(mndx));
  m_scores = squeeze(mean(Q.scores,1));
  [readfcn] = get_readframe_fcn(L.vid2files{mndx});
  
  mpts = zeros(2,5);
  for ii = 1:5
    gg = argmax(m_scores(:,:,ii));
    [gg1,gg2] = ind2sub([size(m_scores,1),size(m_scores,2)],gg);
    mpts(2,ii) = gg1; 
    mpts(1,ii) = gg2;
  end
  
  ff = readfcn(100);
  pndx = find(L.expidx==mndx,1);
  figure(ndx);
  subplot(2,3,1); 
  imshow(ff); hold on;
  scatter(squeeze(L.pts(1,2,:,pndx)),squeeze(L.pts(2,2,:,pndx)),20,cc,'.'); 
  scatter(mpts(1,:)*scale,mpts(2,:)*scale,20,cc,'x'); 
  hold off;
  for ii = 1:5
    subplot(2,3,ii+1); 
    imagesc(m_scores(:,:,ii)); axis equal;
    hold on;
    scatter(mpts(1,ii),mpts(2,ii),20,'xw'); 
    
    hold off;
  end
  
end