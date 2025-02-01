PPDATASAVE.X = pTstT(:,:,:,end);
PPDATASAVE.tblMFT = tblMFtrk;
PPDATASAVE.viewCalibrationData = obj.lObj.viewCalibrationDataCurrent;

d = 2;
nviews = 2;
[N,nsamples,D] = size(PPDATASAVE.X);
npts = D / d / nviews;
PPDATASAVE.X = reshape(PPDATASAVE.X,[N,nsamples,npts,nviews,d]);
mov = obj.lObj.currMovIdx;
PPDATASAVE.movFiles = obj.lObj.movieFilesAll(mov,:);

save(sprintf('ppdata_sh_%d.mat',mov),'-struct','PPDATASAVE');

%%

% load ppdata.mat;
% if ndims(X) >= 5,
%   X = permute(X,[1,2,3,5,4]);
% end

load ppdata_sh_81.mat X tblMFT viewCalibrationData movFiles;
[N,nsamples,npts,nviews,d_in] = size(X);
%mov = unique(tblMFT.mov);
%assert(numel(mov)==1);
%caldata = viewCalibrationData{mov};
caldata = viewCalibrationData;
movFiles = strrep(movFiles,'$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
assert(~isempty(caldata));

projfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4523_gt080618_made20180627.lbl';
dosavefigs = false;
preload = true;






%% SH single-view

npts = 5;
nviews = 1;
nfrms = 1455;
ntgts = 1;
d = 2;

VIEW = 1;
if VIEW==1
  EXPDIR = 'f:\kbPostProc20180924\C001H001S0009';
  HMAPDIR = fullfile(EXPDIR,'C001H001S0009_c_trn20180924T200537_20181008T130634_hmap');
  HMAPNR = 350;
  HMAPNC = 230;
  MOVFILE = fullfile(EXPDIR,'C001H001S0009_c.avi');  
else
  EXPDIR = 'f:\kbPostProc20180924\C002H001S0009';
  HMAPDIR = fullfile(EXPDIR,'C002H001S0009_c_trn20180924T200537_20181008T130634_hmap');
  HMAPNR = 350;
  HMAPNC = 350;
  MOVFILE = fullfile(EXPDIR,'C002H001S0009_c.avi');
end

DOSAVEFIGS = false;

% this is how the parameters should be
viewsindependent = true;
pointsindependent = true;

% don't know how to set these
viterbi_poslambda = 1;
viterbi_misscost = 5;
viterbi_dampen = .25;

preload = true;

tblMFT = table((1:nfrms)','variablenames',{'frm'});

pp = PostProcess();
%pp.SetUseGeometricError(true);
%pp.SetCalibrationData(caldata);
%pp.SetSampleData(X,viewsindependent,pointsindependent);

hmr = HeatmapReader;
hmr.init(HMAPDIR,HMAPNR,HMAPNC,nfrms,npts,ntgts);
readscorefuns = cell(npts,nviews);
for pti = 1:npts
  readscorefuns{pti} = @(n) hmr.read(n,pti,1,'normalize',true);
end
scales = ones(nviews,d);
pp.SetHeatmapData(readscorefuns,nfrms,scales);
pp.SetMovieFiles({MOVFILE},preload);
pp.SetMFTInfo(tblMFT);
%pp.EstimateKDESigma();

%%
lblfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4879_gtcomplete_cacheddata.lbl';
lbl = load(lblfile,'-mat');
t = Labeler.lblFileGetLabels(lblfile);
%%
mfa2 = FSPath.fullyLocalizeStandardize(lbl.movieFilesAll(:,2),lbl.projMacros);
imovUn = unique(t.mov);
mfa2lbled = mfa2(imovUn);

%%
starttime = tic;
pp.SetJointSamples(false);
pp.SetAlgorithm('maxdensity');
pp.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));

starttime = tic;
pp.SetAlgorithm('median');
pp.run(); pp.postdata.viterbi_grid.x = xyBestTraj;
fprintf('Time to run median: %f\n',toc(starttime));

postdata_singleheatmap = pp.postdata;

starttime = tic;
pp.SetAlgorithm('viterbi');
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,...
  'misscost',inf);  
pp.run('force',true);
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata_singleheatmap.viterbvlag1IPTokxi_indep_nomiss = pp.postdata.viterbi_indep;

% starttime = tic;
% pp.SetAlgorithm('viterbi');
% pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,...
%   'misscost',viterbi_misscost);
% pp.run();
% fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
% postdata_singleheatmap.viterbi_indep_miss = pp.postdata.viterbi_indep;
% 




%% SH full run: generate hmap dirs
SWEEPROOTDIR = '/groups/branson/bransonlab/apt/tmp/postproc/sh/ppsweep/';
HMAPDIR = fullfile(SWEEPROOTDIR,'..','hmap');
MOVLIST = fullfile(SWEEPROOTDIR,'..','shtrn719_vw2_movs.txt');
HMAPDIRLIST = fullfile(SWEEPROOTDIR,'..','shtrn719_vw2_hmapdirs.txt');

fh = fopen(HMAPDIRLIST,'w');
movs = readtxtfile(MOVLIST);
nmovs = numel(movs);
fprintf('%d movs\n',nmovs);
for imov=1:nmovs
  mov = movs{imov};
  [movP,movF,movE] = fileparts(mov);
  hmdir = fullfile(HMAPDIR,movP,[movF '_hmap']);
  if exist(hmdir,'dir')==0
    warningNoTrace('imov=%d; hmdir DNE',imov);
  end
  
  fprintf(fh,'%s\n',hmdir);
end
fclose(fh);

%% SH full run
% lblfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4879_gtcomplete_cacheddata.lbl';
% ld = load(lblfile,'-mat');

[movP,movF,movE]= fileparts(movs{1});
hmdir = fullfile(HMAPDIR,movP,[movF '_hmap']);

nowstr = datestr(now,'yyyymmddTHHMMSS');
savefile = sprintf('runpp_%s.mat',nowstr);

allppobj = RunPostProcessing_HeatmapData2(hmdir,...
  'nviews',1,...
  'npts',5,...
  'heatmap_lowthresh',0,...
  'heatmap_highthresh',1,...
  'heatmap_nsamples',150,...
  'heatmap_sample_algorithm','gmm',...
  'usegeometricerror',true,...
  'kde_sigma_px',5,...
  'viterbi_poslambda',.0027,...
  'viterbi_misscost',inf,...
  'viterbi_dampen',.25,...
  'viterbi_grid_acradius',12,...
  'savefile',savefile,...
  'ncores',4);

%% SH full run params
s = struct();
s.npts = 5;
s.heatmap_lowthresh = 0;
s.heatmap_highthresh = 1;
s.heatmap_nsamples = 150;
s.viterbi_poslambda = .0027;
s.viterbi_misscost = inf;
s.viterbi_dampen = .25;
s.viterbi_grid_acradius = 12;
save('ppprmsSH.mat','-struct','s');
%% SH full run
allppobj = RunPostProcessing_HeatmapData2(...
  'rootdir','/groups/branson/bransonlab/apt/tmp/postproc/sh/ppsweep',...
  'paramfiles','ppbase.mat',...
  'algorithms',{'viterbi_grid'});



%% Bub heatmap data 

lblfile = fullfile('/groups/branson/home/robiea/Projects_data/Labeler_APT',...
  'Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl');
ld = load(lblfile,'-mat');

hmdirs = mydir('/groups/branson/home/kabram/temp/alice/umdn_trks','name','.*_hmap','isdir',true);
hmtype = 'jpg';

ihmdirs2run = 5;
iflies2run = 9;
startframe = 17000;
endframe = 18550;

heatmap_lowthresh = 0.025;
heatmap_hithresh = 1;
heatmap_nsamples = 150;

viterbi_dampen = 0.25;
viterbi_poslambda = 0.0027;
viterbi_misscost = inf;

allpostdata = cell(1,numel(hmdirs));
allpp = cell(1,numel(hmdirs));
for hmdiri = ihmdirs2run
  hmdir = hmdirs{hmdiri};

  [~,fname] = fileparts(hmdir);
  expname = fname(1:end-5);
  
  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{moviei};
  trxfile = strrep(trxfile,'$movdir',expdir);
  
  td = load(trxfile);
  nflies = numel(td.trx);
  allpostdata{moviei} = cell(1,nflies);
  allpp{moviei} = cell(1,nflies);
  
  npts = ld.cfg.NumLabelPoints;
  
  for fly = iflies2run
    savefile = fullfile('ppres20181019',sprintf('%s_trx_%d.mat',expname,fly));
    fprintf('Movie %d / %d = %s, fly %d / %d...\n',moviei,numel(hmdirs),expname,fly,nflies);
    
    if exist(savefile,'file'),
      rd = load(savefile);
      allpostdata{moviei}{fly} = rd.postdata;
      allpp{moviei}{fly} = rd.pp;
    else
%      frames = unique(t_labeled{moviei}(fly_labeled{moviei}==fly&pti_labeled{moviei}==1&di_labeled{moviei}==1));
%       frames = frms2run;
%       if isempty(frames),
%         continue;
%       end

      scorefilenames = get_score_filenames(hmdir,fly,1,startframe:endframe,...
        'hmtype',hmtype);
      if ~(all(cellfun(@exist,scorefilenames(:))>0)),
        warning('Not all heatmap files exist, skipping hmdiri = %d, fly = %d...\n',hmdiri,fly);
        continue;
      end
      
      [allpostdata{moviei}{fly},allpp{moviei}{fly}] = ...
        RunPostProcessing_HeatmapData(hmdir,...
        'hmtype',hmtype,...
        'lblfile',lblfile,...
        'targets',fly,...
        'startframe',startframe,...
        'endframe',endframe,... %'frames',frames,...
        'pts2run',12,...
        'viterbi_poslambda',viterbi_poslambda,...
        'viterbi_misscost',viterbi_misscost,...
        'viterbi_dampen',viterbi_dampen,...
        'algorithms',{'maxdensity' 'median' 'viterbi'},...
        'savefile',savefile,...
        'heatmap_lowthresh',heatmap_lowthresh,...
        'heatmap_highthresh',heatmap_hithresh,...
        'heatmap_nsamples',heatmap_nsamples,...
        'heatmap_sample_algorithm','gmm');      
    end
  end
end

%% Bub: generate MFT pch dirs

FRMCLOSERAD = 75;
LBLFILE = '/groups/branson/bransonlab/apt/tmp/postproc/bub/ppsweep/multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
PCHDIR = '/groups/branson/bransonlab/apt/tmp/postproc/bub/ppsweep/mftsbub';
DRY = '/groups/branson/bransonlab/apt/tmp/postproc/bub/ppsweep/mftsbub.dry';

diary(DRY);

tLbl = Labeler.lblFileGetLabels(LBLFILE);
fprintf('%d labeled rows\n',height(tLbl));
nmov = 5;
%frms2pp = cell(nmov,1);
for imov=1:nmov  
  tf = tLbl.mov==imov;
  tsub = tLbl(tf,:);
  tgts = unique(tsub.iTgt);
  maxtgt = max(tgts);
  %frms2pp{imov} = cell(maxtgt,1);
  for itgt=1:maxtgt
    if ~ismember(itgt,tgts)
      continue;
    end
    
    tsubsub = tsub(tsub.iTgt==itgt,:);
    nfrmorig = height(tsubsub);
    frmtf = false(1,max(tsubsub.frm)+FRMCLOSERAD);
    frmtf(tsubsub.frm) = true;
    frmtf = imdilate(frmtf,ones(1,2*FRMCLOSERAD+1));
    frmnew = find(frmtf);
    [sp,ep] = get_interval_ends(frmtf);
    ep = ep-1;
    fprintf('imov itgt nfrmsold nfrmsnew nints %d %d %d %d %d\n',imov,itgt,...
      nfrmorig,numel(frmnew),numel(ep));
    
    for isp=1:numel(sp)
      
      spI = sp(isp);
      epI = ep(isp);
      fnameS = sprintf('imov%02d_itgt%02d_sfrm%06d_nfrm%06d.m',imov,itgt,...
        spI,epI-spI+1);
      fname = fullfile(PCHDIR,fnameS);
      
      fh = fopen(fname,'w');
      fprintf(fh,'lblfileImov = %d\n',imov);
      fprintf(fh,'targets = %d\n',itgt);
      fprintf(fh,'startframe = %d\n',spI);
      fprintf(fh,'endframe = %d\n',epI);
      fclose(fh);
      
      fprintf('Wrote %s\n',fnameS);
      
    end
    %frms2pp{imov}{itgt} = struct('frm0',tsubsub.frm,'frm1',frmnew,'frm1sp',sp,'frm1ep',ep);
  end
end

diary off
%% Bub

lblfile = fullfile('/groups/branson/home/robiea/Projects_data/Labeler_APT',...
  'Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl');
ld = load(lblfile,'-mat');

hmdirs = mydir('/groups/branson/home/kabram/temp/alice/umdn_trks','name','.*_hmap','isdir',true);
hmtype = 'jpg';

ihmdirs2run = 5;
hmdir = hmdirs{ihmdirs2run};

nowstr = datestr(now,'yyyymmddTHHMMSS');
savefile = sprintf('runpp_%s.mat',nowstr);

allppobj = RunPostProcessing_HeatmapData2(hmdir,...
  'lblfile',lblfile,... % used to specify a trxfile
  'lblfileImov',1,... % movie index, use this trxfile
  'npts',[],... %  'pts2run',14,...
  'targets',1,...
  'startframe',3371,...
  'endframe',3385,...
  'heatmap_lowthresh',.025,...
  'heatmap_highthresh',1,...
  'heatmap_nsamples',150,...
  'heatmap_sample_algorithm','gmm',...
  'usegeometricerror',true,...
  'kde_sigma_px',5,...
  'viterbi_poslambda',.0027,...
  'viterbi_misscost',inf,...
  'viterbi_dampen',.25,...
  'viterbi_grid_acradius',12,...
  'savefile',savefile,...
  'ncores',4);

%% Bub
s = struct();
s.lblfile = 'multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
s.lblfilehmdirs = hmdirs([2 1 5 3 4])';
save('ppbase.mat','-struct','s');

s = struct();
s.targets = 1;
s.startframe = 3371;
s.endframe = 3375;
s.heatmap_lowthresh = .025;
s.heatmap_highthresh = 1;
s.heatmap_nsamples = 150;
s.viterbi_poslambda = .0027;
s.viterbi_misscost = inf;
s.viterbi_dampen = .25/2;
s.viterbi_grid_acradius = 12;
save('ppprms_vdamp_dn.mat','-struct','s');


%% Bub
allppobj = RunPostProcessing_HeatmapData2('foo',...
  'rootdir','/groups/branson/bransonlab/apt/tmp/postproc/bub/ppsweep',...
  'paramfiles','ppbase.mat#ppprms_vdamp_dn.mat#mftsbub/imov01_itgt01_sfrm003346_nfrm000170.m',...
  'algorithms',{'viterbi_grid'});


%% GT: sh

postdata = pp.postdata;

if VIEW==1
  lpos = loadSingleVariableMatfile('lpos164_vw1ptsonly.mat');
else
  lpos = loadSingleVariableMatfile('lpos164_vw2ptsonly.mat');
  lpos = SparseLabelArray.full(lpos{1});
end

frmsGT = find(all(~isnan(reshape(lpos,10,1455)),1));
fprintf(1,'%d frmsGT.\n',numel(frmsGT));

pd = postdata;

algsAll = fieldnames(pd);
lposGT = lpos(:,:,frmsGT); % npts x d x nGT
lposGT = permute(lposGT,[3 1 2]);

dxyAll = cell(0,1);
for alg=algsAll(:)',alg=alg{1}; %#ok<FXSET>
  %[n x npts x (x/y) x nviews x nsets]
  tpos = pd.(alg).x(frmsGT,:,:); % nGT x npts x d
  
  % XXX hand-massage crop
  tpos(:,:,1) = tpos(:,:,1) + 1 - 1;
  tpos(:,:,2) = tpos(:,:,2) + 121 - 1;
  
  dxyAll{end+1,1} = tpos-lposGT;
end
dxyAll = cat(5,dxyAll{:});

[hFig,hAxs] = GTPlot.ptileCurves(dxyAll,...
  'ptiles',[50 75 90],...
  'setNames',algsAll...
  );


%% GT: bub add GV
hmBestTrajXY = hmBestTrajPQ(:,[2 1]);
%hmBestTrajXY(1501:1551,:) = nan;
xyBestTraj = PostProcess.UntransformByTrx(hmBestTrajXY,pp.trx,pp.heatmap_origin);
pp.postdata.viterbi_grid_lam0.x = nan(1551,17,2);
pp.postdata.viterbi_grid_lam0.x(:,12,:) = xyBestTraj;
%% GT: bub


% delta_train = struct;
% delta_test = struct;
% for i = 1:numel(algorithms),
%   algorithm = algorithms{i};
%   tmpidx = find(all(all(~isnan(predpose_train.(algorithm)),2),3));
%   delta_train.(algorithm) = truepose_train(tmpidx,:,:)-predpose_train.(algorithm)(tmpidx,:,:);
%   tmpidx = find(all(all(~isnan(predpose_test.(algorithm)),2),3));
%   delta_test.(algorithm) = truepose_test(tmpidx,:,:)-predpose_test.(algorithm)(tmpidx,:,:);
% end
% 
% err_train = struct;
% err_test = struct;
% for i = 1:numel(algorithms),
%   algorithm = algorithms{i};
%   err_train.(algorithm) = sqrt(sum(delta_train.(algorithm).^2,3));
%   err_test.(algorithm) = sqrt(sum(delta_test.(algorithm).^2,3));
% end
% 
% figure(17);
% clf;
% algorithm = algorithms{1};
% maxv = prctile([abs(delta_test.(algorithm)(:));abs(delta_train.(algorithm)(:))],99);
% subplot(2,2,1);
% imagesc(delta_train.(algorithm)(:,:,1));
% set(gca,'CLim',[-1,1]*maxv);
% title('Train, x');
% subplot(2,2,2);
% imagesc(delta_train.(algorithm)(:,:,2));
% set(gca,'CLim',[-1,1]*maxv);
% title('Train, y');
% subplot(2,2,3);
% imagesc(delta_test.(algorithm)(:,:,1));
% set(gca,'CLim',[-1,1]*maxv);
% title('Test, x');
% subplot(2,2,4);
% imagesc(delta_test.(algorithm)(:,:,2));
% set(gca,'CLim',[-1,1]*maxv);
% title('Test, y');
% colormap(myredbluecmap);
% colorbar;
% 
% figure(18);
% clf;
% algcolors = lines(numel(algorithms));
% colors = jet(npts);
% markers = {'-o','-s','-+'};
% h = [];
% nrax = ceil(sqrt(npts));
% ncax = ceil(npts/nrax);
% hax = createsubplots(nrax,ncax,.05);
% for i = 1:npts,
%   hold(hax(i),'on');
%   title(hax(i),sprintf('Landmark %d',i));
%   for j = 1:numel(algorithms),
%     h(j) = plot(hax(i),sort(err_test.(algorithms{j})(:,i)),markers{j},'Color',algcolors(j,:));
%   end
% end
% 
% legend(h,algorithms,'Interpreter','none');
% 
% delete(hax(npts+1:end));
% 
% % compute percentiles of errors per part
% prctiles_compute = [50,75,90,95,97.5,99];
% errprctile_train = struct;
% errprctile_test = struct;
% for i = 1:numel(algorithms),
%   algorithm = algorithms{i};
%   
%   errprctile_train.(algorithm) = prctile(err_train.(algorithm),prctiles_compute,1);
%   errprctile_test.(algorithm) = prctile(err_test.(algorithm),prctiles_compute,1);
%   
% end


%% compare kdeweights to hm mag
ipt = pp.pts2run;
assert(isscalar(ipt),'for now');
kdeweights = pp.kdedata.indep(:,:,ipt);
xyreps = squeeze(pp.sampledata.x(:,:,ipt,:));
xyrepsTrx = pp.TransformByTrx(xyreps,pp.trx,pp.heatmap_origin);

[n,nrep] = size(kdeweights);
hmmags = nan(size(kdeweights));
for i=1:n
  disp(i);
  hm = pp.ReadHeatmapScore(12,1,i);
  for irep=1:nrep
    r = xyrepsTrx(i,irep,2);
    r = round(r);
    c = xyrepsTrx(i,irep,1);
    c = round(c);
    hmmags(i,irep) = hm(r,c);
  end
end

figure;
scatter(kdeweights(:),hmmags(:));
figure;
scatter(hmmags(:),kdeweights(:));

%% Massage pp.kdedata.indep
pp.kdedata.indep(:,:,12) = hmmags;

save cx_JHS_K_85321_CsChr_RigD_20150909T163219_trx_9_kdedataindepReplacedbyHMMags.mat pp

%% rerun viterbi
pp.postdata.viterbi_indep_orig = pp.postdata.viterbi_indep;
pp.RunViterbi;
save cx_JHS_K_85321_CsChr_RigD_20150909T163219_trx_9_kdedataindepReplacedbyHMMags.mat pp

%% export trks
pd = pp.postdata;
algs = fieldnames(pd);
for alg=algs(:)',alg=alg{1};
  ptrk = pd.(alg).x;
  ptrk = permute(ptrk,[2 3 1]);
  trk = TrkFile(ptrk,'pTrkFrm',pp.tblMFT.frm,'pTrkiTgt',9); 
  trkname = [alg '.trk'];
  trk.save(trkname);
  fprintf('Saved %s\n',trkname); 
end
  
%% Set up lObj/import trk
trkfileMain = 'maxdensity_indep.trk';
trkfileLpos2 = 'viterbi_indep.trk';
ptf = lObj.tracker;
tv = ptf.trkVizer;

mIdx = lObj.currMovIdx;
ptf.trackResAddTrkfile(mIdx,{trkfileMain});
ptf.newLabelerMovie;
% import trkfileLpos2 manually

tv.heatMapEnable = 1;
hmd = hmdirs{5};
tv.heatMapInit(hmd,[],[]);
tv.heatMapIPtsShow = 12;
tv.heatMapRawImType = 'invert';
hmr = tv.heatMapReader;
hmr.hmnormalizeType = 'lohi';
hmr.hmlothresh = 128;
hmr.hmhithresh = 255;

lObj.movieCenterOnTargetLandmark = true;
lObj.movieCenterOnTargetIpt = 12;

%% visualize kdesamps with weights (AC)
frm = lObj.currFrame;
isamp = find(frm==pp.tblMFT.frm);
if isempty(isamp)
  error('frame %d is not in pp.',frm);
end
assert(isscalar(isamp));

ipt = pp.pts2run;
assert(isscalar(ipt),'for now');
kdeweights = pp.kdedata.indep(isamp,:,ipt);
kdeweights = kdeweights(:);
reps = squeeze(pp.sampledata.x(isamp,:,ipt,:));

ax = lObj.gdata.axes_curr;
deleteValidGraphicsHandles(hS);
hS = scatter(ax,reps(:,1),reps(:,2),kdeweights*1e3,'r');




%% Motion model Spatial


NR = 180;
NC = 180;
DOWNSAMP = 3;
NRDS = ceil(NR/DOWNSAMP);
NCDS = ceil(NC/DOWNSAMP);
vsum = zeros(NR,NC,2,npts2run);
vcnt = zeros(NR,NC,npts2run);
vsumds = zeros(NRDS,NCDS,2,npts2run);
vcntds = zeros(NRDS,NCDS,npts2run);
for iipt=1:npts2run
for i=1:size(x,1)
  ipt = pp.pts2run(iipt);
  
  xy = x(i,ipt,:);
  r = round(xy(2)); % when istrx, untransform can de-integerize
  c = round(xy(1));
  vtmp = v(i,ipt,:); % dxy that takes us to next frame
  vsum(r,c,:,iipt) = vsum(r,c,:,iipt)+vtmp;
  vcnt(r,c,iipt) = vcnt(r,c,iipt)+1;
  
  rds = ceil(r/DOWNSAMP);
  cds = ceil(c/DOWNSAMP);
  vsumds(rds,cds,:,iipt) = vsumds(rds,cds,:,iipt)+vtmp;
  vcntds(rds,cds,iipt) = vcntds(rds,cds,iipt)+1;
end
end
%%
LENSCALEFAC = 3;
LINEWIDTHSCALEFAC = 0.25;
vmean = vsum./reshape(vcnt,NR,NC,1,npts2run);
vmeands = vsumds./reshape(vcntds,NRDS,NCDS,1,npts2run);

MINCOUNTTOPLOT = 2;
vmeanplot = vmeands;
vcntplot = vcntds;
nrplot = NRDS;
ncplot = NCDS;

hFig = figure(11);
clf;
axsz = ceil(sqrt(npts2run));
axs = mycreatesubplots(axsz,axsz,[.1,.05]);
clrs = lines(npts2run);
for iipt=1:npts2run
  ipt = pp.pts2run(iipt);
  ax = axs(iipt);
  axes(ax);
  hold(ax,'on');
  
  for r=1:nrplot
  for c=1:ncplot
    if vcntplot(r,c,iipt)>=MINCOUNTTOPLOT
      v = vmeanplot(r,c,:,iipt);
      plot(ax,[c c+v(1)*LENSCALEFAC],[r r+v(2)*LENSCALEFAC],'-',...
        'color',clrs(iipt,:),...
        'linewidth',vcntplot(r,c,iipt)*LINEWIDTHSCALEFAC);
      plot(ax,c+v(1)*LENSCALEFAC,r+v(2)*LENSCALEFAC,'.','color',clrs(iipt,:),'markersize',10);
    end    
  end
  end
  
  axis ij
  grid on;
  title(num2str(ipt),'fontweight','bold');
end

linkaxes(axs);
  




%%
hFig = figure(22);
clf;
ax = axes;
hold(ax,'on');
weightsdisp = round(kdeweights*800);
maxweight = max(weightsdisp)
clrs = jet(maxweight);
for i=1:size(reps,1)
  plot(ax,reps(i,1),reps(i,2),'.','markersize',10,'Color',clrs(weightsdisp(i,:),:));
end



%%


assert(tv.heatMapNoRawIm);
assert(isscalar(tv.heatMapIPtsShow));
im = lObj.gdata.images_all.CData;
size(im)
ptclr = tv.ptClrs(tv.heatMapIPtsShow,:);
im = im./reshape(ptclr,1,1,3); 
im = im(:,:,1);
idximreps = sub2ind(size(im),round(reps(:,2)),round(reps(:,1)));
kdeweightsHM = im(idximreps);

figure;
scatter(kdeweightsHM,kdeweights);
grid on


%% Massage pp.kdedata.indep
% 
% kdesamps0 = pp.kdedata.indep;
% assert(isequaln(squeeze(pp.sampledata.x_in),pp.sampledata.x));
% 
% [nfrm,nsamp,npt] = size(kdesamps);
% for f=1:nfrm
%   if mod(f,10)==0, disp(f); end
%   for iipt=1:npts2run
%     ipt = pp.pts2run(iipt);
%     hm = pp.ReadHeatmapScore(ipt,1,f);
%     for isamp=1:nsamp
%       sampxy = pp.sampledata.x_in(f,isamp,ipt,1,:);
%       
%       
%       % xxx react
%       sampxyRel = PostProcess.TransformByTrx(sampxy,pp.trx,pp.radius_trx,f);
%       
%       sampr = round(sampxyRel(2));
%       sampc = round(sampxyRel(1));
%       
%       if isnan(sampr) || isnan(sampc)
%         hmsamp = 0;
%       else
%         hmsamp = hm(sampr,sampc);
%       end
%       
%       kdesamps(f,isamp,ipt) = hmsamp; % in [0,1], normalized hm val
%     end
%   end
% end
% 
% %pp.kdedata.indep = kdesamps;


%%
figure(19);
clf;
hold on;
prctilecolors = lines(numel(prctiles_compute));
if numel(algorithms)==1,
  off = 0;
else
  off = linspace(-.2,.2,numel(algorithms));
end
htest = nan(1,numel(prctiles_compute));
for i = 1:numel(algorithms),
  algorithm = algorithms{i};
  for j = 1:numel(prctiles_compute),
    htrain = plot((1:npts)+off(i),errprctile_train.(algorithm)(j,:),'+','LineWidth',2,'Color',prctilecolors(j,:));
    htest(j) = plot((1:npts)+off(i),errprctile_test.(algorithm)(j,:),'o','LineWidth',2,'Color',prctilecolors(j,:));
  end
end
legends = cell(1,numel(prctiles_compute));
for j = 1:numel(prctiles_compute),
  legends{j} = [num2str(prctiles_compute(j)),'%ile'];
end
legends{end} = [num2str(prctiles_compute(j)),'th %ile, test error'];
legend([htest,htrain],[legends,{'Train error'}],'Location','northwest');
xticks = [off+1,2:npts];
shortalgorithms = {'maxdensity';'median';'viterbi'};
xticklabels = [shortalgorithms;cellstr(num2str((2:npts)'))];
set(gca,'XTick',xticks,'XTickLabels',xticklabels,'XTickLabelRotation',90);
xlabel('Landmark');

%%






n = 100;

if DOSAVEFIGS,
hfig = pp.PlotReprojectionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

savefig(hfig,sprintf('ppfigs/ReprojectionSamples_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReprojectionSamples_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReprojectionSamples_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');

hfig = pp.PlotReconstructionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
savefig(hfig,sprintf('ppfigs/ReconstructionSamples_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReconstructionSamples_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReconstructionSamples_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');

hfig = pp.PlotSampleScores();
savefig(hfig,sprintf('ppfigs/ReconstructionSampleScores_mov%d.fig',mov));
saveas(hfig,sprintf('ppfigs/ReconstructionSampleScores_mov%d.png',mov),'png');
print(hfig,sprintf('ppfigs/ReconstructionSampleScores_mov%d.pdf',mov),'-dpdf','-fillpage');

end

hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
if dosavefigs,
savefig(hfig,sprintf('ppfigs/ReprojectionMaxDensityJoint_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReprojectionMaxDensityJoint_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReprojectionMaxDensityJoint_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');
end

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
savefig(hfig,sprintf('ppfigs/ReprojectionMedian_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReprojectionMedian_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReprojectionMedian_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
if dosavefigs,
savefig(hfig,sprintf('ppfigs/ReprojectionMaxDensityIndep_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReprojectionMaxDensityIndep_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReprojectionMaxDensityIndep_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');
end

postdata = pp.postdata;

starttime = tic;
pp.SetJointSamples(true);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_joint_nomiss: %f\n',toc(starttime));
postdata.viterbi_joint_nomiss = pp.postdata.viterbi_joint;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

if dosavefigs,
savefig(hfig,sprintf('ppfigs/ReprojectionViterbiJointNoMiss_mov%d_n%d.fig',mov,n));
saveas(hfig,sprintf('ppfigs/ReprojectionViterbiJointNoMiss_mov%d_n%d.png',mov,n),'png');
print(hfig,sprintf('ppfigs/ReprojectionViterbiJointNoMiss_mov%d_n%d.pdf',mov,n),'-dpdf','-fillpage');
end

starttime = tic;
pp.SetJointSamples(true);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_joint_miss: %f\n',toc(starttime));
postdata.viterbi_joint_miss = pp.postdata.viterbi_joint;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = pp.postdata.viterbi_indep;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);


starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = pp.postdata.viterbi_indep;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

x_perview_joint = pp.sampledata.x_perview;
w_joint = pp.sampledata.w;
kde_sigma = pp.kde_sigma;

%%

hfig = figure(8);
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');

%% test out other parameters, even though they are wrong - joint views

% this is how the parameters should be
viewsindependent = false;
pointsindependent = false;

% don't know how to set these
viterbi_poslambda = 1;
viterbi_misscost = 5;
viterbi_dampen = .25;

% resample according to weights
x_perview_joint_resample = nan(size(x_perview_joint));
nsamples = size(x_perview_joint,2);
for i = 1:size(x_perview_joint,1),
  
  wcurr = w_joint(i,:);
  idxcurr = randsample(1:nsamples,nsamples,true,wcurr);
  x_perview_joint_resample(i,:,:,:,:) = x_perview_joint(i,idxcurr,:,:,:);
  
end

%% 
pp = PostProcess();
pp.SetUseGeometricError(true);
pp.SetCalibrationData(caldata);
pp.SetSampleData(x_perview_joint_resample,viewsindependent,pointsindependent);
pp.SetMovieFiles(movFiles,preload);
pp.SetMFTInfo(tblMFT);
pp.SetKDESigma(kde_sigma);

starttime = tic;
pp.SetJointSamples(true);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));

n = 100;

hfig = pp.PlotReprojectionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
hfig = pp.PlotReconstructionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
hfig = pp.PlotSampleScores();
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

postdata = pp.postdata;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_joint_nomiss: %f\n',toc(starttime));
postdata.viterbi_joint_nomiss = pp.postdata.viterbi_joint;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_joint_miss: %f\n',toc(starttime));
postdata.viterbi_joint_miss = pp.postdata.viterbi_joint;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

%%

hfig = figure;
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');

%% test out other parameters, even though they are wrong - multiview, independent points

% this is how the parameters should be
viewsindependent = true;
pointsindependent = true;

% don't know how to set these
viterbi_poslambda = 1;
viterbi_misscost = 5;
viterbi_dampen = .25;

%% 
pp = PostProcess();
pp.SetUseGeometricError(true);
pp.SetCalibrationData(caldata);
pp.SetSampleData(X,viewsindependent,pointsindependent);
pp.SetMovieFiles(movFiles,preload);
pp.SetMFTInfo(tblMFT);
pp.EstimateKDESigma();

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_indep: %f\n',toc(starttime));

n = 100;

hfig = pp.PlotReprojectionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
hfig = pp.PlotReconstructionSamples(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
hfig = pp.PlotSampleScores();
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

postdata = pp.postdata;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = pp.postdata.viterbi_indep;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = pp.postdata.viterbi_indep;
hfig = pp.PlotReprojection(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

%%

hfig = figure;
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');

%% test out other parameters, even though they are wrong - singleview, joint points

% this is how the parameters should be
viewsindependent = true;
pointsindependent = false;

% don't know how to set these
viterbi_poslambda = .01;
viterbi_misscost = 5;
viterbi_dampen = .25;
viewi = 1;
kde_sigma_px = 5;

%% 
pp = PostProcess();
pp.SetUseGeometricError(true);
%pp.SetCalibrationData(caldata);
pp.SetSampleData(X(:,:,:,viewi,:),viewsindependent,pointsindependent);
pp.SetMovieFiles(movFiles(viewi),preload);
pp.SetMFTInfo(tblMFT);
pp.SetKDESigma(kde_sigma_px);

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_indep: %f\n',toc(starttime));

n = 100;
hfig = pp.PlotSampleDistribution(n,'plotkde',true);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

pp.SetJointSamples(true);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

postdata = pp.postdata;

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = pp.postdata.viterbi_indep;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = pp.postdata.viterbi_indep;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);



starttime = tic;
pp.SetJointSamples(true);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_joint_nomiss: %f\n',toc(starttime));
postdata.viterbi_joint_nomiss = pp.postdata.viterbi_joint;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.SetJointSamples(true);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_joint_miss: %f\n',toc(starttime));
postdata.viterbi_joint_miss = pp.postdata.viterbi_joint;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);



%%

hfig = figure;
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');

%% test out other parameters, even though they are wrong - singleview, indep points

% this is how the parameters should be
viewsindependent = true;
pointsindependent = true;

% don't know how to set these
viterbi_poslambda = .01;
viterbi_misscost = 5;
viterbi_dampen = .25;
viewi = 2;
kde_sigma_px = 5;

%% 
pp = PostProcess();
pp.SetUseGeometricError(true);
%pp.SetCalibrationData(caldata);
pp.SetSampleData(X(:,:,:,viewi,:),viewsindependent,pointsindependent);
pp.SetMovieFiles(movFiles(viewi),preload);
pp.SetMFTInfo(tblMFT);
pp.SetKDESigma(kde_sigma_px);

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_indep: %f\n',toc(starttime));

n = 100;
hfig = pp.PlotSampleDistribution(n,'plotkde',true);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);
hfig = PlotSampleScores(pp,'plotkde',true);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

postdata = pp.postdata;

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = pp.postdata.viterbi_indep;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);

starttime = tic;
pp.SetJointSamples(false);
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = pp.postdata.viterbi_indep;
hfig = pp.PlotTimepoint(n);
set(hfig,'Units','pixels','Position',[1,25,2560,1469]);



%%

hfig = figure;
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');


%% make some heatmap data out of sample data

d = d_in;

maxx = reshape(max(max(max(X,[],1),[],2),[],3),[nviews,d]);
%sig = min(maxx(:))/30;
sig = 3;
imsz = ceil(fliplr(maxx)+2*sig);

heatmapims = cell(1,nviews);
for viewi = 1:nviews,
  heatmapims{viewi} = zeros([imsz(viewi,:),npts,N]);
end

hsize = ceil(2.5*sig);
fil = fspecial('gaussian',hsize*2+1,sig);
for n = 1:N,
  for viewi = 1:nviews,
    for pti = 1:npts,
      
      for samplei = 1:nsamples,
        
        p = squeeze(round(X(n,samplei,pti,viewi,:)));
        heatmapims{viewi}(p(2),p(1),pti,n) = heatmapims{viewi}(p(2),p(1),pti,n) + 1;
        
      end
      
      heatmapims{viewi}(:,:,pti,n) = imfilter(heatmapims{viewi}(:,:,pti,n),fil,0,'same');

    end
  end  
end

maxv = nan(1,nviews);
for viewi = 1:nviews,
  maxv(viewi) = max(heatmapims{viewi}(:));
  heatmapims{viewi} = heatmapims{viewi}/maxv(viewi);
end

clf;
hax = createsubplots(1,nviews);
colors = lines(npts);
him = nan(1,nviews);
for n = 1:N,
  for viewi = 1:nviews,
    
    imrgb = reshape(min(1,sum(heatmapims{viewi}(:,:,:,n).*reshape(colors,[1,1,npts,3]),3)),[imsz(viewi,:),3]);
    if n == 1,
      him(viewi) = image(imrgb,'Parent',hax(viewi));
      axis(hax(viewi),'image');
    else
      set(him(viewi),'CData',imrgb);
    end
  end
  drawnow;
end

readscorefuns = cell(npts,nviews);
for viewi = 1:nviews,
  for pti = 1:npts,
    readscorefuns{pti,viewi} = @(n) heatmapims{viewi}(:,:,pti,n);
  end
end

%% run heatmap data through algorithms

nviews = size(readscorefuns,2);
scales = ones(nviews,2);
pp = PostProcess();
pp.SetUseGeometricError(true);
pp.SetCalibrationData(caldata);
pp.SetHeatmapData(readscorefuns,N,scales);
pp.SetMovieFiles(movFiles,preload);
pp.SetMFTInfo(tblMFT);
pp.EstimateKDESigma();

starttime = tic;
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_indep: %f\n',toc(starttime));

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));

postdata = pp.postdata;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = pp.postdata.viterbi_indep;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run('force',true);
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = pp.postdata.viterbi_indep;

%% compare sample and heatmap results

hfig = figure;
clf;
[N,K,npts,d] = size(pp.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2,'Color',colors(i,:));
      hold(hax(ipt,id),'on');
      if isfield(postdata0,fns{i}),
        hs(i) = plot(hax(ipt,id),1:N,postdata0.(fns{i}).x(:,ipt,id),':','LineWidth',2,'Color',.3+.7*colors(i,:));
      end
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend([h,hs],[cellfun(@(x) ['heatmap, ',x],fns,'Uni',0),cellfun(@(x) ['sample, ',x],fns(isfield(postdata0,fns)),'Uni',0)],'Interpreter','none');
linkaxes(hax,'x');

%% single view heatmap

nviews = 1;
scales = ones(nviews,d);
pp = PostProcess();
pp.SetUseGeometricError(true);
pp.SetCalibrationData(caldata);
pp.SetHeatmapData(readscorefuns(:,1),N,scales);
pp.SetMovieFiles(movFiles(1),preload);
pp.SetMFTInfo(tblMFT);
pp.SetKDESigma(kde_sigma_px);

starttime = tic;
pp.algorithm = 'maxdensity';
pp.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));

starttime = tic;
pp.algorithm = 'median';
pp.run();
fprintf('Time to run median: %f\n',toc(starttime));

postdata_singleheatmap = pp.postdata;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
pp.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata_singleheatmap.viterbi_indep_nomiss = pp.postdata.viterbi_indep;

starttime = tic;
pp.algorithm = 'viterbi';
pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
pp.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata_singleheatmap.viterbi_indep_miss = pp.postdata.viterbi_indep;

%% find labels

traintestsplitfile = '~kabram/temp/alice/umdn_trks/splitdata.json';
ttinfo = jsondecode(fileread(traintestsplitfile));
trainidx = ttinfo{1};
testidx = ttinfo{2};
movieidxtrain = trainidx(:,1)+1;
ttrain = trainidx(:,2)+1;
flytrain = trainidx(:,3)+1;
movieidxtest = testidx(:,1)+1;
ttest = testidx(:,2)+1;
flytest = testidx(:,3)+1;

nmovies = numel(ld.labeledpos);
pti_labeled = cell(1,nmovies);
di_labeled = cell(1,nmovies);
t_labeled = cell(1,nmovies);
fly_labeled = cell(1,nmovies);
truepose_train = nan([numel(ttrain),npts,d]);

for mi = 1:nmovies,
  [pti_labeled{mi},di_labeled{mi},t_labeled{mi},fly_labeled{mi}] = ind2sub(ld.labeledpos{mi}.size,ld.labeledpos{mi}.idx);
end
for mi = 1:nmovies,
  ldrow = [fly_labeled{mi}(pti_labeled{mi}==1&di_labeled{mi}==1),t_labeled{mi}(pti_labeled{mi}==1&di_labeled{mi}==1)];
  ttrow = [[flytrain(movieidxtrain==mi);flytest(movieidxtest==mi)],[ttrain(movieidxtrain==mi);ttest(movieidxtest==mi)]];
  assert(all(ismember(ldrow,ttrow,'rows')) && all(ismember(ttrow,ldrow,'rows')));

  for i = find(movieidxtrain(:)'==mi),
    idxcurr = t_labeled{mi} == ttrain(i) & fly_labeled{mi} == flytrain(i);
    for pti = 1:npts,
      for di = 1:d,
        j = find(idxcurr & pti_labeled{mi}==pti & di_labeled{mi}==di);
        assert(numel(j)==1);
        truepose_train(i,pti,di) = ld.labeledpos{mi}.val(j);
      end
    end
  
  end
end

truepose_test = nan([numel(ttest),npts,d]);
for mi = 1:nmovies,
  for i = find(movieidxtest(:)'==mi),
    idxcurr = t_labeled{mi} == ttest(i) & fly_labeled{mi} == flytest(i);
    for pti = 1:npts,
      for di = 1:d,
        j = find(idxcurr & pti_labeled{mi}==pti & di_labeled{mi}==di);
        assert(numel(j)==1);
        truepose_test(i,pti,di) = ld.labeledpos{mi}.val(j);
      end
    end
  
  end
end

truepose_test_allframes = cell(1,nmovies);
for mi = 1:nmovies,
  % not sure how general this is
  nfliescurr = ld.labeledpos{mi}.size(4);
  truepose_test_allframes{mi} = cell(1,nfliescurr);
  nframescurr = ld.labeledpos{mi}.size(3);
  for fi = 1:nfliescurr,
    truepose_test_allframes{mi}{fi} = nan([nframescurr,npts,d]);
    idxcurr = find(movieidxtest==mi & flytest==fi);
    for i = idxcurr(:)',
      truepose_test_allframes{mi}{fi}(ttest(i),:,:) = truepose_test(i,:,:);
    end
  end
end

truepose_train_allframes = cell(1,nmovies);
for mi = 1:nmovies,
  % not sure how general this is
  nfliescurr = ld.labeledpos{mi}.size(4);
  truepose_train_allframes{mi} = cell(1,nfliescurr);
  nframescurr = ld.labeledpos{mi}.size(3);
  for fi = 1:nfliescurr,
    truepose_train_allframes{mi}{fi} = nan([nframescurr,npts,d]);
    idxcurr = find(movieidxtrain==mi & flytrain==fi);
    for i = idxcurr(:)',
      truepose_train_allframes{mi}{fi}(ttrain(i),:,:) = truepose_train(i,:,:);
    end
  end
end


%% real heatmap data - flies

lblfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
ld = load(lblfile,'-mat');

hmdirs = mydir('/groups/branson/home/kabram/temp/alice/umdn_trks','name','.*_hmap','isdir',true);
hmtype = 'jpg';
%hmdirs = mydir('/groups/branson/home/bransonk/tracking/code/APT/data/cx_hmaps','name','cx.*_hmap','isdir',true);
%hmtype = 'mjpg';
%hmdir = '/groups/branson/home/kabram/temp/alice/umdn_trks/cx_GMR_SS00030_CsChr_RigC_20150826T144616_hmap';

viterbi_poslambda = 0.0027;
viterbi_misscost = .02;


allpostdata = cell(1,numel(hmdirs));
allpp = cell(1,numel(hmdirs));
for hmdiri = 1:numel(hmdirs),
  hmdir = hmdirs{hmdiri};

  [~,fname] = fileparts(hmdir);
  expname = fname(1:end-5);
  
  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{moviei};
  trxfile = strrep(trxfile,'$movdir',expdir);
  
  td = load(trxfile);
  nflies = numel(td.trx);
  allpostdata{moviei} = cell(1,nflies);
  allpp{moviei} = cell(1,nflies);
  
  npts = ld.cfg.NumLabelPoints;
  
  for fly = 1:nflies,
    savefile = fullfile('ppresults_frames',sprintf('%s_trx_%d.mat',expname,fly));
    fprintf('Movie %d / %d = %s, fly %d / %d...\n',moviei,numel(hmdirs),expname,fly,nflies);
    
    if exist(savefile,'file'),
      rd = load(savefile);
      allpostdata{moviei}{fly} = rd.postdata;
      allpp{moviei}{fly} = rd.pp;
    else
      continue;
      frames = unique(t_labeled{moviei}(fly_labeled{moviei}==fly&pti_labeled{moviei}==1&di_labeled{moviei}==1));
      if isempty(frames),
        continue;
      end

      scorefilenames = get_score_filenames(hmdir,fly,1,frames,'hmtype',hmtype,'firstframe',td.trx(fly).firstframe);
      if ~(all(cellfun(@exist,scorefilenames(:))>0)),
        warning('Not all heatmap files exist, skipping hmdiri = %d, fly = %d...\n',hmdiri,fly);
        continue;
      end
      
      [allpostdata{moviei}{fly},allpp{moviei}{fly}] = ...
        RunPostProcessing_HeatmapData(hmdir,'lblfile',lblfile,'targets',fly,'startframe',td.trx(fly).firstframe,'endframe',td.trx(fly).endframe,...
        'savefile',savefile,'hmtype',hmtype,'frames',frames,'viterbi_poslambda',viterbi_poslambda,'viterbi_misscost',viterbi_misscost);
    end
  end
end

%%
% viterbi_poslambda = .01;
% 
% 
% fly = 1;
% nviews = ld.cfg.NumViews;
% npts = ld.cfg.NumLabelPoints;
% realN = ld.movieInfoAll{moviei}.nframes;
% d = 2;
% startframe = 1;
% N = td.trx(fly).nframes;
% 
% trxcurr = td.trx(fly);
% fns = fieldnames(trxcurr);
% i0 = startframe-trxcurr.firstframe+1;
% i1 = startframe+N-1-trxcurr.firstframe+1;
% for i = 1:numel(fns),
%   if numel(trxcurr.(fns{i})) == 1 || ischar(trxcurr.(fns{i})),
%     continue;
%   end
%   l = trxcurr.nframes-numel(trxcurr.(fns{i}));
%   trxcurr.(fns{i}) = trxcurr.(fns{i})(i0:i1-l);
% end
% 
% 
% readscorefuns = cell(npts,nviews);
% for viewi = 1:nviews,
%   for pti = 1:npts,
%     readscorefuns{pti,viewi} = get_readscore_fcn(hmdir,fly,pti);
%   end
% end
% 
% kde_sigma_px = 5;
% 
% scales = ones(nviews,d);
% pp = PostProcess();
% pp.SetUseGeometricError(true);
% pp.SetHeatmapData(readscorefuns,N,scales,trxcurr);
% pp.SetMovieFiles({moviefile});
% pp.SetKDESigma(kde_sigma_px);
% pp.heatmap_lowthresh = .1;
% pp.heatmap_highthresh = .5;
% 
% starttime = tic;
% pp.algorithm = 'maxdensity';
% pp.run();
% fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));
% % 
% % hfig = 1;
% % for n = 1:N,
% %   pp.PlotTimepoint(n,'hfig',hfig,'markerparams',{'LineWidth',2});
% %   axis([td.trx(fly).x(n+startframe-1)+pp.radius_trx(1)*[-1,1],td.trx(fly).y(n+startframe-1)+pp.radius_trx(2)*[-1,1]]);
% %   drawnow;
% % end
% 
% starttime = tic;
% pp.algorithm = 'median';
% pp.run();
% fprintf('Time to run median: %f\n',toc(starttime));
% % 
% % hfig = 1;
% % for n = 1:N,
% %   pp.PlotTimepoint(n,'hfig',hfig,'markerparams',{'LineWidth',2});
% %   axis([td.trx(fly).x(n+startframe-1)+pp.radius_trx(1)*[-1,1],td.trx(fly).y(n+startframe-1)+pp.radius_trx(2)*[-1,1]]);
% %   drawnow;
% % end
% 
% postdata_singleheatmap = pp.postdata;
% 
% starttime = tic;
% pp.algorithm = 'viterbi';
% pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',inf);
% pp.run();
% fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
% postdata_singleheatmap.viterbi_indep_nomiss = pp.postdata.viterbi_indep;
% 
% hfig = 1;
% for n = 1:N,
%   pp.PlotTimepoint(n,'hfig',hfig,'markerparams',{'LineWidth',2});
%   axis([td.trx(fly).x(n+startframe-1)+pp.radius_trx(1)*[-1,1],td.trx(fly).y(n+startframe-1)+pp.radius_trx(2)*[-1,1]]);
%   drawnow;
% end
% 
% starttime = tic;
% pp.algorithm = 'viterbi';
% pp.SetViterbiParams('poslambda',viterbi_poslambda,'dampen',viterbi_dampen,'misscost',viterbi_misscost);
% pp.run();
% fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
% postdata_singleheatmap.viterbi_indep_miss = pp.postdata.viterbi_indep;
% 
% %% plot results over time
% 
% hfig = figure;
% clf;
% N = pp.N;
% npts = pp.npts;
% d = 2;
% 
% hax = createsubplots(npts,d,[.02,.02;.03,.01]);
% hax = reshape(hax,[npts,d]);
% fns = fieldnames(postdata);
% colors = lines(numel(fns));
% h = nan(1,numel(fns));
% 
% for ipt = 1:npts,
%   for id = 1:d,
%     for i = 1:numel(fns),
%       h(i) = plot(hax(ipt,id),1:N,postdata_singleheatmap.(fns{i}).x(:,ipt,id),'LineWidth',2,'Color',colors(i,:));
%       hold(hax(ipt,id),'on');
%     end
%     %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
%     axisalmosttight([],hax(ipt,id));
%     box(hax(ipt,id),'off');
%   end
% end
% set(hax,'XLim',[0,N+1]);%,'YLim',[min(pp.sampledata.x(:)),max(pp.sampledata.x(:))]);
% set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
% legend(h,fns,'Interpreter','none');
% linkaxes(hax,'x');

%% compute error 

for i = 1:numel(allpostdata),
  if ~isempty(allpostdata{i}) && ~isempty(allpostdata{i}{1}),
    d = size(allpostdata{i}{1}.maxdensity_indep.x,3);
    algorithms = fieldnames(allpostdata{i}{1});
    break;
  end
end

predpose_test = struct;
for i = 1:numel(algorithms),
  predpose_test.(algorithms{i}) = nan(size(truepose_test));
end

for mi = 1:nmovies,
  if isempty(allpostdata{mi}),
    continue;
  end
  for fly = 1:numel(allpostdata{mi}),
    if isempty(allpostdata{mi}{fly}),
      continue;
    end
    %firstframe = td.trx(fly).firstframe;
    firstframe = allpp{mi}{fly}.trx.firstframe;
    idxcurr = find(movieidxtest==mi & flytest == fly);
    if isempty(idxcurr),
      continue;
    end
    
    for j = 1:numel(algorithms),
      algorithm = algorithms{j};
      
      ts = ttest(idxcurr);
      predpose_test.(algorithm)(idxcurr,:,:) = allpostdata{mi}{fly}.(algorithm).x(ts-firstframe+1,:,:);
      
    end
    
  end
end

predpose_train = struct;
for i = 1:numel(algorithms),
  predpose_train.(algorithms{i}) = nan(size(truepose_train));
end

for mi = 1:nmovies,
  if isempty(allpostdata{mi}),
    continue;
  end
  for fly = 1:numel(allpostdata{mi}),
    if isempty(allpostdata{mi}{fly}),
      continue;
    end
    %firstframe = td.trx(fly).firstframe;
    firstframe = allpp{mi}{fly}.trx.firstframe;
    idxcurr = find(movieidxtrain==mi & flytrain == fly);
    if isempty(idxcurr),
      continue;
    end
    
    for j = 1:numel(algorithms),
      algorithm = algorithms{j};
      
      ts = ttrain(idxcurr);
      predpose_train.(algorithm)(idxcurr,:,:) = allpostdata{mi}{fly}.(algorithm).x(ts-firstframe+1,:,:);
      
    end
    
  end
end




%% 

traintestsplitfile = '~kabram/temp/alice/umdn_trks/splitdata.json';
ttinfo = jsondecode(fileread(traintestsplitfile));
trainidx = ttinfo{1};
testidx = ttinfo{2};
movieidxtrain = trainidx(:,1)+1;
ttrain = trainidx(:,2)+1;
flytrain = trainidx(:,3)+1;
movieidxtest = testidx(:,1)+1;
ttest = testidx(:,2)+1;
flytest = testidx(:,3)+1;

movieidx = cat(1,movieidxtrain,movieidxtest);
ts = cat(1,ttrain,ttest);
flies = cat(1,flytrain,flytest);


lblfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
ld = load(lblfile,'-mat');
macrofns = fieldnames(ld.projMacros);

trxfiles = cell(1,numel(ld.movieFilesAll));
for i = 1:numel(ld.movieFilesAll),
  moviefile = ld.movieFilesAll{i};
  
  for j = 1:numel(macrofns),
    moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{i};
  trxfile = strrep(trxfile,'$movdir',expdir);
  trxfiles{i} = trxfile;
end
intervals2track = GetIntervalsToTrackForErrorMeasurement(movieidx,flies,ts,trxfiles,'winrad',100);

%% viterbi parameter sweep

% parameters to change:
% kde_sigma_px
% viterbi_poslambda
% viterbi_misscost
% viterbi_dampen

%viterbi_misscosts_try = linspace(3.5,4.5,20);

viterbi_poslambda = 0.0027;
viterbi_misscosts_try = [.1,.15];
%viterbi_misscosts_try = [linspace(3.5,10,10),1,2,3];
%viterbi_poslambda = .01;
viterbi_dampen = .25;
script = '/groups/branson/home/bransonk/tracking/code/APT/RunPostProcessing_HeatmapData/for_redistribution_files_only/run_RunPostProcessing_HeatmapData.sh';
clusterinfo = GetMATLABClusterInfo();
assert(exist(clusterinfo.MCR,'dir')>0);

nparams = numel(viterbi_misscosts_try);

ppdir = '/groups/branson/home/bransonk/tracking/code/APT/ppgrid_20180920';
if ~exist(ppdir,'dir'),
  mkdir(ppdir);
end
[~,token] = fileparts(ppdir);

lblfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
ld = load(lblfile,'-mat');
nmovies = numel(hmdirs);
ncores = 1;

jobinfos = [];

for hmdiri = 1:numel(hmdirs),
  hmdir = hmdirs{hmdiri};

  [~,fname] = fileparts(hmdir);
  expname = fname(1:end-5);
  
  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{moviei};
  trxfile = strrep(trxfile,'$movdir',expdir);
  
  td = load(trxfile);
  nflies = numel(td.trx);
  
  for parami = 1:nparams,
    
    for fly = 1:nflies,
      
      frames = unique(t_labeled{moviei}(fly_labeled{moviei}==fly&pti_labeled{moviei}==1&di_labeled{moviei}==1));
      if isempty(frames),
        continue;
      end
      
      ppdircurr = fullfile(ppdir,sprintf('param%02d',parami));
      if ~exist(ppdircurr,'dir'),
        mkdir(ppdircurr);
      end
                  
      savefile = fullfile(ppdircurr,sprintf('results_%s_trx_%d.mat',expname,fly));
      paramsfile = fullfile(ppdircurr,sprintf('params_%s_trx_%d.mat',expname,fly));
      argstr = sprintf('%s paramsfile %s',hmdir,paramsfile);
      
      jobid = sprintf('p%d_m%d_f%d_%s',parami,moviei,fly,token);
      shfile = fullfile(ppdircurr,sprintf('run_movie%d_fly%d.sh',moviei,fly));
      logfile = fullfile(ppdircurr,sprintf('run_movie%d_fly%d.out',moviei,fly));
      
      jobinfocurr = struct;
      jobinfocurr.token = token;
      jobinfocurr.moviei = moviei;
      jobinfocurr.fly = fly;
      jobinfocurr.parami = parami;
      jobinfocurr.paramsfile = paramsfile;
      jobinfocurr.jobid = jobid;
      jobinfocurr.shfile = shfile;
      jobinfocurr.logfile = logfile;
      jobinfocurr.savefile = savefile;
  
      jobinfos = structappend(jobinfos,jobinfocurr);
      
      if exist(savefile,'file'),
        fprintf('File %s exists, skipping\n',savefile);
        continue;
      end
      
      params = struct;
      params.hmtype = 'jpg';
      params.viterbi_misscost = viterbi_misscosts_try(parami);
      params.viterbi_poslambda = viterbi_poslambda;
      params.viterbi_dampen = viterbi_dampen;
      params.algorithms = {'viterbi'};
      
      params.lblfile = lblfile;
      params.targets = fly;
      params.startframe = td.trx(fly).firstframe;
      params.endframe = td.trx(fly).endframe;
      params.savefile = savefile;
      params.frames = frames;
      params.ncores = 0;
      
      save(paramsfile,'-struct','params');
      MakeMATLABClusterScript(shfile,jobid,script,argstr,'MCR',clusterinfo.MCR,'TMP_ROOT_DIR',clusterinfo.TMP_ROOT_DIR);
      
      cmd = sprintf('bsub -n %d -R"affinity[core(1)]" -o %s -J %s %s',ncores,logfile,jobid,shfile);
      cmd2 = sprintf('ssh login1 ''source /etc/profile; cd %s; %s''',pwd,cmd);
      unix(cmd2);
      fprintf('%s\n',cmd2);
    end
  end
end

save(sprintf('JobInfo_%s.mat',token),'jobinfos');

%% also run maxdensity and median

script = '/groups/branson/home/bransonk/tracking/code/APT/RunPostProcessing_HeatmapData/for_redistribution_files_only/run_RunPostProcessing_HeatmapData.sh';
clusterinfo = GetMATLABClusterInfo();
assert(exist(clusterinfo.MCR,'dir')>0);

ppdir = '/groups/branson/home/bransonk/tracking/code/APT/ppmaxmed_20180830';
if ~exist(ppdir,'dir'),
  mkdir(ppdir);
end
[~,token] = fileparts(ppdir);

lblfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_xv7.lbl';
ld = load(lblfile,'-mat');
nmovies = numel(hmdirs);
ncores = 1;

jobinfos_maxmed = [];

for hmdiri = 1:numel(hmdirs),
  hmdir = hmdirs{hmdiri};

  [~,fname] = fileparts(hmdir);
  expname = fname(1:end-5);
  
  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{moviei};
  trxfile = strrep(trxfile,'$movdir',expdir);
  
  td = load(trxfile);
  nflies = numel(td.trx);
  
  for fly = 1:nflies,
    
    frames = unique(t_labeled{moviei}(fly_labeled{moviei}==fly&pti_labeled{moviei}==1&di_labeled{moviei}==1));
    if isempty(frames),
      continue;
    end
    
    ppdircurr = fullfile(ppdir,sprintf('param%02d',parami));
    if ~exist(ppdircurr,'dir'),
      mkdir(ppdircurr);
    end
    
    savefile = fullfile(ppdircurr,sprintf('results_%s_trx_%d.mat',expname,fly));
    paramsfile = fullfile(ppdircurr,sprintf('params_%s_trx_%d.mat',expname,fly));
    argstr = sprintf('%s paramsfile %s',hmdir,paramsfile);
    
    jobid = sprintf('p%d_m%d_f%d_%s',parami,moviei,fly,token);
    shfile = fullfile(ppdircurr,sprintf('run_movie%d_fly%d.sh',moviei,fly));
    logfile = fullfile(ppdircurr,sprintf('run_movie%d_fly%d.out',moviei,fly));
    
    jobinfocurr = struct;
    jobinfocurr.token = token;
    jobinfocurr.moviei = moviei;
    jobinfocurr.fly = fly;
    jobinfocurr.parami = parami;
    jobinfocurr.paramsfile = paramsfile;
    jobinfocurr.jobid = jobid;
    jobinfocurr.shfile = shfile;
    jobinfocurr.logfile = logfile;
    jobinfocurr.savefile = savefile;
    
    jobinfos_maxmed = structappend(jobinfos_maxmed,jobinfocurr);
    
    if exist(savefile,'file'),
      fprintf('File %s exists, skipping\n',savefile);
      continue;
    end
    
    params = struct;
    params.hmtype = 'jpg';
    params.algorithms = {'maxdensity','median'};
    
    params.lblfile = lblfile;
    params.targets = fly;
    params.startframe = td.trx(fly).firstframe;
    params.endframe = td.trx(fly).endframe;
    params.savefile = savefile;
    params.frames = frames;
    params.ncores = 0;
    
    save(paramsfile,'-struct','params');
    MakeMATLABClusterScript(shfile,jobid,script,argstr,'MCR',clusterinfo.MCR,'TMP_ROOT_DIR',clusterinfo.TMP_ROOT_DIR);
    
    cmd = sprintf('bsub -n %d -R"affinity[core(1)]" -o %s -J %s %s',ncores,logfile,jobid,shfile);
    cmd2 = sprintf('ssh login1 ''source /etc/profile; cd %s; %s''',pwd,cmd);
    unix(cmd2);
    fprintf('%s\n',cmd2);
  end
end

%% store firstframe for each fly

firstframes = cell(1,numel(hmdirs));
for hmdiri = 1:numel(hmdirs),

  hmdir = hmdirs{hmdiri};

  [~,fname] = fileparts(hmdir);
  expname = fname(1:end-5);
  
  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end
  
  expdir = fileparts(moviefile);
  trxfile = ld.trxFilesAll{moviei};
  trxfile = strrep(trxfile,'$movdir',expdir);
  
  td = load(trxfile);
  nflies = numel(td.trx);
  
  firstframes{moviei} = [td.trx.firstframe];
  
end

%% load in results

allpostdata_grid = cell(nparams,numel(hmdirs));
for parami = 1:nparams,
  for moviei = 1:numel(hmdirs),
    allpostdata_grid{parami,moviei} = cell(1,numel(firstframes{moviei}));
  end
end
  
for i = 1:numel(jobinfos),
  jobinfocurr = jobinfos(i);
  if isempty(allpostdata_grid{jobinfocurr.parami,jobinfocurr.moviei}),
    allpostdata_grid{jobinfocurr.parami,jobinfocurr.moviei} = {};
  end
  if ~exist(jobinfocurr.savefile,'file'),
    fprintf('File %d = %s does not exist, skipping\n',i,jobinfocurr.savefile);
    continue;
  end
  sd = load(jobinfocurr.savefile,'postdata','timestamp');
  allpostdata_grid{jobinfocurr.parami,jobinfocurr.moviei}{jobinfocurr.fly} = sd.postdata;
  jobinfos(i).timestamp = sd.timestamp;
end

%% load in max and median results

for i = 1:numel(jobinfos_maxmed),
  jobinfocurr = jobinfos_maxmed(i);
  if ~exist(jobinfocurr.savefile,'file'),
    fprintf('File %d = %s does not exist, skipping\n',i,jobinfocurr.savefile);
    continue;
  end
  if isempty(allpostdata_grid{parami,jobinfocurr.moviei}{jobinfocurr.fly}),
    continue;
  end
  sd = load(jobinfocurr.savefile,'postdata','timestamp');
  fns = fieldnames(sd.postdata);
  for parami = 1:nparams,
    for j = 1:numel(fns),
      allpostdata_grid{parami,jobinfocurr.moviei}{jobinfocurr.fly}.(fns{j}) = sd.postdata.(fns{j});
    end
  end
end

%% compute error

truedata = struct(...
  'trainidx',trainidx,...
  'testidx',testidx,...
  'movieidxtrain',movieidxtrain,...
  'ttrain',ttrain,...
  'flytrain',flytrain,...
  'movieidxtest',movieidxtest,...
  'ttest',ttest,...
  'flytest',flytest,...
  'truepose_train',truepose_train,...
  'truepose_test',truepose_test);

parami = 1;
err_train_grid = struct;
err_test_grid = struct;
err_train_stats_grid = struct;
err_test_stats_grid = struct;
isfirst = true;

statfns = {'prctiles_perpart'
  'prctiles_worstpart'
  'prctiles_avepart'
  };

for parami = 1:nparams,
  [err_train_curr,err_test_curr,err_train_stats_curr,err_test_stats_curr] = ComputeTrainTestError(allpostdata_grid(parami,:),firstframes,truedata);
  if isfirst,
    algorithms = fieldnames(err_train_curr);
    err_train_grid = err_train_curr;
    err_test_grid = err_test_curr;
    err_train_stats_grid = err_train_stats_curr;
    err_test_stats_grid = err_test_stats_curr;
    isfirst = false;
  else
    for i = 1:numel(algorithms),
      algorithm = algorithms{i};
      err_train_grid.(algorithm)(:,:,parami) = err_train_curr.(algorithm);
      err_test_grid.(algorithm)(:,:,parami) = err_test_curr.(algorithm);
      for j = 1:numel(statfns),
        err_train_stats_grid.(algorithm).(statfns{j})(:,:,parami) = err_train_stats_curr.(algorithm).(statfns{j});
        err_test_stats_grid.(algorithm).(statfns{j})(:,:,parami) = err_test_stats_curr.(algorithm).(statfns{j});
      end
    end
  end
end

hfig = 1238;
figure(hfig);
clf;

param_colors = jet(nparams+2)*.7;
off = linspace(-.3,.3,nparams+2);
h = plot(err_test_stats_grid.viterbi_miss_indep.prctiles_compute'+off(1:end-2),squeeze(err_test_stats_grid.viterbi_miss_indep.prctiles_worstpart),'o');
hold on;
hmax = plot(err_test_stats_grid.maxdensity_indep.prctiles_compute+off(end-1),squeeze(err_test_stats_grid.maxdensity_indep.prctiles_worstpart(:,:,1)),'s');
hmed = plot(err_test_stats_grid.median.prctiles_compute+off(end),squeeze(err_test_stats_grid.median.prctiles_worstpart(:,:,1)),'s');
legends = cell(1,nparams+2);
for i = 1:nparams,
  legends{i} = sprintf('Viterbi, miss cost = %.2f',viterbi_misscosts_try(i));
  set(h(i),'Color','w','MarkerFaceColor',param_colors(i,:));
end
set(hmax,'Color','w','MarkerFaceColor',param_colors(end-1,:));
set(hmed,'Color','w','MarkerFaceColor',param_colors(end,:));

legends{nparams+1} = 'maxdensity';
legends{nparams+2} = 'median';
legend([h;hmax;hmed],legends,'Location','NorthWest');
xlabel('Percentile')
ylabel('Worst part error (px)')


hfig = 1239;
figure(hfig);
clf;

param_colors = jet(nparams+2)*.7;
off = linspace(-.3,.3,nparams+2);
h = plot(err_test_stats_grid.viterbi_miss_indep.prctiles_compute'+off(1:end-2),squeeze(err_test_stats_grid.viterbi_miss_indep.prctiles_avepart),'o');
hold on;
hmax = plot(err_test_stats_grid.maxdensity_indep.prctiles_compute+off(end-1),squeeze(err_test_stats_grid.maxdensity_indep.prctiles_avepart(:,:,1)),'s');
hmed = plot(err_test_stats_grid.median.prctiles_compute+off(end),squeeze(err_test_stats_grid.median.prctiles_avepart(:,:,1)),'s');
legends = cell(1,nparams+2);
for i = 1:nparams,
  legends{i} = sprintf('Viterbi, miss cost = %.2f',viterbi_misscosts_try(i));
  set(h(i),'Color','w','MarkerFaceColor',param_colors(i,:));
end
set(hmax,'Color','w','MarkerFaceColor',param_colors(end-1,:));
set(hmed,'Color','w','MarkerFaceColor',param_colors(end,:));

legends{nparams+1} = 'maxdensity';
legends{nparams+2} = 'median';
legend([h;hmax;hmed],legends,'Location','NorthWest');
xlabel('Percentile')
ylabel('Average part error (px)')

%% nvisible

nmiss = cell(size(allpostdata_grid));
nprocess = cell(size(allpostdata_grid));
for i = 1:numel(allpostdata_grid),
  nmiss{i} = zeros(size(allpostdata_grid{i}));
  nprocess{i} = zeros(size(allpostdata_grid{i}));
  for j = 1:numel(allpostdata_grid{i}),
    if isempty(allpostdata_grid{i}{j}) || ~isfield(allpostdata_grid{i}{j},'viterbi_miss_indep'),
      continue;
    end
    nmiss{i}(j) = nnz(allpostdata_grid{i}{j}.viterbi_miss_indep.isvisible==0);
    nprocess{i}(j) = nnz(~isnan(allpostdata_grid{i}{j}.viterbi_miss_indep.sampleidx));
  end
end