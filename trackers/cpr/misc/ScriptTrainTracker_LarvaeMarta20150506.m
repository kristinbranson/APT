% try a variety of parameters

%clear all
doeq = false;
loadH0 = false;
cpr_type = 2;
docacheims = false;
maxNTr = 10000;
winrad = 475;

addpath ..;
addpath ../video_tracking;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

defaultfolder = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
%defaultfile = 'M134_M174_20150423.mat';
defaultfile = 'LarvaMuscles_94A04D_Frames1_20150429_20150506.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  fullfile(defaultfolder,defaultfile));
if isnumeric(file),
  return;
end

[~,savestr] = fileparts(file);

ld = load(fullfile(folder,file));

fgthresh = 1000;

%% "track"

ld.trxfilestr = 'trx.mat';
trx = cell(1,numel(ld.expdirs));
for i = 1:numel(ld.expdirs),
  expdir = ld.expdirs{i};
  [readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
  trx = struct('x',nan(1,nframes),'y',nan(1,nframes),'firstframe',1,'endframe',nframes,'nframes',nframes,'off',0,'bbs',nan(nframes,4));
  for t = 1:nframes,
    im = readframe(t);
    isfore = im >= fgthresh;
    stats = regionprops(isfore,{'Area','Centroid','BoundingBox'});
    [~,j] = max([stats.Area]);
    stats = stats(j);
    trx.x(t) = stats.Centroid(1);
    trx.y(t) = stats.Centroid(2);
    trx.bbs(t,:) = stats.BoundingBox;
  end
  if fid > 0,
    fclose(fid);
  end
  save(fullfile(expdir,ld.trxfilestr),'trx');
end

fprintf('x-width: %d, y-width: %d\n',...
  round(max(max(bbs(:,1)+bbs(:,3)-trx.x'),max(trx.x'-bbs(:,1)))),...
  round(max(max(bbs(:,2)+bbs(:,4)-trx.y'),max(trx.y'-bbs(:,2)))));

%% subsample training data

nTr=min(numel(ld.expidx),maxNTr);
if nTr < numel(ld.expidx),
  idx = SubsampleTrainingDataBasedOnMovement(ld,maxNTr);
  idx = idx(randperm(numel(idx)));
  nTr = numel(idx);
  %idx=randsample(numel(IsTr),nTr);
else
  idx = randperm(nTr);
end

hfig = 2;
figure(hfig);
clf;
nr = 5;
nc = 5;
nplot = nr*nc;
hax = createsubplots(nr,nc,.01);

nexpdirs = numel(ld.expdirs);

i0 = round(linspace(1,nplot+1,nexpdirs+1));

idxsample = [];
for expi = 1:nexpdirs,
  
  idxcurr = idx(ld.expidx(idx)==expi);
  ncurr = i0(expi+1)-i0(expi);
  idxsample(i0(expi):i0(expi+1)-1) = idxcurr(randsample(numel(idxcurr),ncurr));  
  
end

npts = size(ld.pts,1);
colors = jet(npts);

expdirprev = '';
fid = -1;
maxint = 0;
for ii = 1:nplot,
  
  i = idxsample(ii);
  expdir = ld.expdirs{ld.expidx(i)};
  fly = ld.flies(i);
  t = ld.ts(i);
  if ~strcmp(expdir,expdirprev),
    if fid > 0,
      fclose(fid);
    end
    [readframe,~,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
    trxcurr = load_tracks(fullfile(expdir,ld.trxfilestr));
    expdirprev = expdir;
  end
  im = readframe(ld.ts(i));
  isfore = im >= fgthresh;
  maxint = prctile(im(isfore),95);
  imagesc(im,'Parent',hax(ii),[0,maxint]);
  axis(hax(ii),'image');
  hold(hax(ii),'on');
  colormap gray;
  for j = 1:npts,
    plot(hax(ii),ld.pts(j,1,i),ld.pts(j,2,i),'wo','MarkerFaceColor',colors(j,:));
  end
  j = t-trxcurr(fly).firstframe+1;
  xcurr = trxcurr(fly).x(j);
  ycurr = trxcurr(fly).y(j);
  
  ax = [xcurr-winrad,xcurr+winrad,ycurr-winrad,ycurr+winrad];
  axis(hax(ii),ax);
  
end
set(hax,'Color','k','XTick',[],'YTick',[]);

if fid > 0,
  fclose(fid);
end

%% read in the training images

[~,n,e] = fileparts(file);
cachedimfile = fullfile(folder,[n,'_cachedims',e]);
pts_norm = nan(size(ld.pts));

if docacheims && exist(cachedimfile,'file'),
  load(cachedimfile,'IsTr','pts_norm');
else
  
  IsTr = cell(1,numel(ld.expidx));
  for expi = 1:numel(ld.expdirs),
    
    idxcurr = find(ld.expidx == expi);
    fprintf('Reading %d frames from experiment %d / %d: %s\n',numel(idxcurr),expi,numel(ld.expdirs),ld.expdirs{expi});
    
    if isempty(idxcurr),
      continue;
    end
    
    [readframe,nframes,fid,headerinfo] = get_readframe_fcn(fullfile(ld.expdirs{expi},ld.moviefilestr));
    trxcurr = load_tracks(fullfile(expdir,ld.trxfilestr));
    
    for i = 1:numel(idxcurr),
      
      t = ld.ts(idxcurr(i));
      fly = ld.flies(idxcurr(i));
      
      j = t-trxcurr(fly).firstframe+1;
      xcurr = round(trxcurr(fly).x(j));
      ycurr = round(trxcurr(fly).y(j));
      
      pts_norm(:,1,idxcurr(i)) = ld.pts(:,1,idxcurr(i))-xcurr+winrad+1;
      pts_norm(:,2,idxcurr(i)) = ld.pts(:,2,idxcurr(i))-ycurr+winrad+1;

      im = readframe(t);
      ncolors = size(im,3);
      if ncolors > 1,
        im = rgb2gray(im);
      end

      IsTr{idxcurr(i)} = padgrab(im,255,ycurr-winrad,ycurr+winrad,xcurr-winrad,xcurr+winrad);
      
    end
    if fid > 0,
      fclose(fid);
    end
    
  end
  IsTr = IsTr(:);
  if docacheims,
    save('-v7.3',cachedimfile,'IsTr','pts_norm');
  end
end

%% histogram equalization

if doeq
  if loadH0
    [fileH0,folderH0]=uigetfile('.mat');
    load(fullfile(folderH0,fileH0));
  else
    H=nan(2^16,nTr);
    mu = nan(1,nTr);
    for i=1:nTr,
      H(:,i)=imhist(IsTr{idx(i)},2^16);
      mu(i) = mean(IsTr{idx(i)}(:));
    end
    % normalize to brighter movies, not to dimmer movies
    %idxuse = mu >= prctile(mu,75);
    idxuse = true(size(idx));
    H0=median(H(:,idxuse),2);
    H0 = H0/sum(H0)*numel(IsTr{1});
  end
  model1.H0=H0;
  % normalize one video at a time
  for expi = 1:numel(ld.expdirs),
    idxcurr = idx(ld.expidx(idx)==expi);
    if isempty(idxcurr),
      continue;
    end
    bigim = cat(1,IsTr{idxcurr});
    bigimnorm = histeq(bigim,H0);
    IsTr(idxcurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,numel(idxcurr)]),imsz(2));
  end
    
%   for i=1:nTr,
%     IsTr2{idx(i)}=histeq(IsTr{idx(i)},H0);
%   end
end


hfig = 3;
figure(hfig);
clf;
nr = 5;
nc = 5;
nplot = nr*nc;
hax = createsubplots(nr,nc,.01);

for ii = 1:nplot,
  
  i = idxsample(ii);
  im = IsTr{i};
  isfore = im >= fgthresh;
  maxint = prctile(im(isfore),95);
  imagesc(im,'Parent',hax(ii),[0,maxint]);
  axis(hax(ii),'image','off');
  hold(hax(ii),'on');
  for j = 1:npts,
    plot(hax(ii),pts_norm(j,1,i),pts_norm(j,2,i),'wo','MarkerFaceColor',colors(j,:));
    if ii == 1,
      text(pts_norm(j,1,i),pts_norm(j,2,i),sprintf('  %d',j),'Color',colors(j,:),'HorizontalAlignment','left','Parent',hax(ii));
    end
  end  
end


%% save training data

ptskeep = 1:npts;

allPhisTr = reshape(permute(pts_norm,[3,1,2]),[size(pts_norm,3),size(pts_norm,1)*size(pts_norm,2)]);
tmp2 = struct;
tmp2.phisTr = allPhisTr(idx,[ptskeep,npts+ptskeep]);
tmp2.bboxesTr = repmat([1,1,2*winrad+1,2*winrad+1],[numel(idx),1]);
tmp2.IsTr = IsTr(idx);
save(sprintf('TrainData_%s.mat',savestr),'-struct','tmp2');

%% choose neighbors

nneighbors = 5;
ds = zeros(1,npts*(npts-1)/2);
for i = 1:numel(idx),
  x = reshape(allPhisTr(idx(i),:),[nfids,2]);
  ds = ds + pdist(x);
end
ds = ds / numel(idx);
D = squareform(ds);
%D = bsxfun(@rdivide,D,sum(D,1));
D(eye(npts)==1) = nan;
ftr_neighbors = cell(1,npts);
for i = 1:npts,
  [~,order] = sort(D(i,:));
  ftr_neighbors{i} = order(1:nneighbors);  
end


%% train tracker

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'LarvaMuscles1';
params.model_nfids = size(tmp2.phisTr,2)/2;
params.model_d = 2;
params.model_nviews = 1;
params.ftr_type = 8;
params.cascade_depth = 200;
%params.ftr_type = 10;
%params.ftr_gen_radius = 25;
params.ftr_gen_radius = fliplr(round(logspace(log10(25),log10(500),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = true;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.numRot = 5;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));
paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

%npts = size(pts_norm,1);
%X = reshape(pts_norm,[size(pts_norm,1)*size(pts_norm,2),size(pts_norm,3)]);
X = tmp2.phisTr;
Xtemplate = reshape(X(1,:),[params.model_nfids,params.model_d]);
for i = 2:size(X,1),
  [~,xcurr] = procrustes(Xtemplate,reshape(X(i,:),[params.model_nfids,params.model_d]),'Scaling',false,'Reflection',false);
  X(i,:) = xcurr(:);
end
[idxinit,initlocs1] = mykmeans(X,round(params.prunePrm.numInit/params.prunePrm.numRot),'Start','furthestfirst','Maxiter',100);
initlocs = nan([params.prunePrm.numRot,size(initlocs1)]);
thetas = linspace(0,2*pi,params.prunePrm.numRot+1);
thetas = thetas(1:end-1);
for i = 1:size(initlocs1,1),
  
  x0 = reshape(initlocs1(i,:),[params.model_nfids,params.model_d]);
  mu = mean(x0,1);
  for j = 1:params.prunePrm.numRot,
    x = rotatearound(x0,thetas(j),mu);
    initlocs(j,i,:) = x(:);
  end
end
initlocs = reshape(initlocs,[params.prunePrm.numRot*size(initlocs1,1),size(initlocs1,2)]);

tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
if doeq,
  tmp.H0 = H0;
end
tmp.prunePrm.motion_2dto3D = false;
tmp.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-append','-struct','tmp');

%% test tracker on the labeled larva

expdir = ld.expdirs{1};
%fly = ld.flies(1);
firstframe = 1;
endframe = 10;
testresfile = '';
fly = 1;
[phisPr,phisPrAll]=test(expdir,trainresfile,testresfile,'moviefilestr',ld.moviefilestr,...
  'trxfilestr',ld.trxfilestr,'winrad',winrad,'flies',fly,...
  'firstframe',firstframe,'endframe',endframe);

%%


savevideo = false;
fly = 1;

nfids = size(phisPr{1},2)/2;

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
trx = load_tracks(fullfile(expdir,ld.trxfilestr));

fprintf('Initial tracking results\n');

maxint = 20000;
fps = 50;
tplot = 0;


figure(1);
clf;
him = imagesc(im,[0,maxint]);
axis image;
colormap gray;
hold on;
htrx = nan(1,nfids);
hcurr = nan(1,nfids);
colors = jet(nfids);
alphas = linspace(0,.8,tplot+2)';
for i = 1:nfids,
%  htrx(i) = patch(nan(102,1),nan(102,1),colors(i,:),...
%    'EdgeAlpha','interp','EdgeColor',colors(i,:)*.7+.3,'FaceColor','none',...
%    'FaceVertexAlphaData',alphas,'LineWidth',2);   
   htrx(i) = plot(nan,nan,'-','Color',colors(i,:)*.5+.5,'LineWidth',2);
end
for i = 1:nfids,
  hcurr(i) = plot(nan,nan,'.-','LineWidth',8,'Color',colors(i,:)*.7,'MarkerSize',10);
%   hcurr(i) = patch(nan,nan,colors(i,:),...
%     'EdgeAlpha',1,'FaceColor','none','LineWidth',3,'EdgeColor',colors(i,:)*.7);
  %hcurr(i) = text(nan,nan,num2str(i),'Color',colors(i,:),'HorizontalAlignment','center','VerticalAlignment','middle');
end
htext = text(.5,.5,'0 s','HorizontalAlignment','left','VerticalAlignment','top','Color',[.8,0,.8],'FontSize',36);
hax = gca;
set(hax,'FontSize',24,'Color','k');

p = phisPr{fly};

fil = normpdf(-6:6,0,2);
fil = fil / sum(fil);
p = imfilter(p,fil','same','replicate','corr');

ax = [.5,.5,size(im,2)+.5,size(im,1)+.5];
border = 20;
axis off;
truesize;

[~,n] = fileparts(expdir);
if savevideo,
  vidobj = VideoWriter(sprintf('TrackingResults_%s_NoTrx2.avi',savestr));
  open(vidobj);
end

gfdata = [];
for t = 1:endframe-firstframe+1,
  
  [im,timestamp] = readframe(firstframe+t-1);
  imsz = size(im);
  set(him,'CData',im);
  for i = 1:nfids,
    
%     set(htrx(i),'XData',[p(max(1,t-tplot):t,i);nan],...
%         'YData',[p(max(1,t-tplot):t,i+nfids);nan],...
%         'FaceVertexAlphaData',alphas(max(1,tplot+2-t):end));
    set(htrx(i),'XData',p(t:-1:max(1,t-tplot),i),'YData',p(t:-1:max(1,t-tplot),i+nfids));
%    set(hcurr(i),'XData',p(t,i),'YData',p(t,nfids+i));
    if mod(i,2) == 1,
      if i < nfids,
        set(hcurr(i),'XData',p(t,[i,i+1]),'YData',p(t,[nfids+i,nfids+i+1]));
      else
        set(hcurr(i),'XData',p(t,[i,1]),'YData',p(t,[nfids+i,nfids+1]));
      end
    end
    %set(hcurr(i),'Position',[p(t,i),p(t,nfids+i),0]);
  end

%   offt = firstframe+t-1-trx(fly).firstframe+1;
%   xcurr = [trx(fly).bbs(offt,1),trx(fly).bbs(offt,1)+trx(fly).bbs(offt,3)];
%   ycurr = [trx(fly).bbs(offt,2),trx(fly).bbs(offt,2)+trx(fly).bbs(offt,4)];
%   
%   if any(xcurr - border < ax(1) | ...
%       xcurr + border > ax(2) | ...
%       ycurr - border < ax(3) | ...
%       ycurr + border > ax(4)),
%     
%     ax = [xcurr(2)-2*winrad,xcurr(2)+2*winrad,ycurr(2)-2*winrad,ycurr(2)+2*winrad];
%     set(hax,'XLim',ax(1:2),'YLim',ax(3:4));
%     set(htext,'Position',[ax(1)+5,ax(3)+5,0]);
%   end
  set(htext,'String',sprintf('%.3fs',t/fps));
  
  %pause(.25);
  drawnow;

  if savevideo,
    if isempty(gfdata),
      gfdata = getframe_initialize(hax);
      fr = getframe_invisible(hax);
      gfdata.sz = size(fr);
    end
    fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
    writeVideo(vidobj,fr);
  end
  
end
if savevideo,
  close(vidobj);
end

if fid > 0,
  fclose(fid);
end

%% test on a new larva

% tmpdir = '/groups/branson/bransonlab/projects/LarvalMuscles/15-04-29/94A04_D_v2';
% files = mydir(fullfile(tmpdir,'Frames_2*tif'));
% for i = 1:numel(files),
%   
%   newname = strrep(files{i},'Frames_2','Frames_1');
%   unix(sprintf('mv %s %s',files{i},newname));
%   
% end

expdir = '/groups/branson/bransonlab/projects/LarvalMuscles/15-04-29/94A04_C';

% "track"

trx = cell(1,numel(ld.expdirs));
[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
trx = struct('x',nan(1,nframes),'y',nan(1,nframes),'firstframe',1,'endframe',nframes,'nframes',nframes,'off',0,'bbs',nan(nframes,4));
for t = 1:nframes,
  fprintf('Frame %d / %d\n',t,nframes);
  im = readframe(t);
  isfore = im >= fgthresh;
  stats = regionprops(isfore,{'Area','Centroid','BoundingBox'});
  [~,j] = max([stats.Area]);
  stats = stats(j);
  trx.x(t) = stats.Centroid(1);
  trx.y(t) = stats.Centroid(2);
  trx.bbs(t,:) = stats.BoundingBox;
end
if fid > 0,
  fclose(fid);
end
save(fullfile(expdir,ld.trxfilestr),'trx');

fprintf('x-width: %d, y-width: %d\n',...
  round(max(max(bbs(:,1)+bbs(:,3)-trx.x'),max(trx.x'-bbs(:,1)))),...
  round(max(max(bbs(:,2)+bbs(:,4)-trx.y'),max(trx.y'-bbs(:,2)))));


%%

%fly = ld.flies(1);
firstframe = 1;
endframe = 100;
testresfile = '';
fly = 1;
[phisPr2,phisPrAll2]=test(expdir,trainresfile,testresfile,'moviefilestr',ld.moviefilestr,...
  'trxfilestr',ld.trxfilestr,'winrad',winrad,'flies',fly,...
  'firstframe',firstframe,'endframe',endframe);

%%

fly = 1;
nfids = size(phisPr2{1},2)/2;

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
trx = load_tracks(fullfile(expdir,ld.trxfilestr));

fprintf('Initial tracking results\n');

maxint = 20000;
fps = 50;
tplot = 50;


figure(1);
clf;
him = imagesc(im,[0,maxint]);
axis image;
colormap gray;
hold on;
htrx = nan(1,nfids);
hcurr = nan(1,nfids);
colors = jet(nfids);
alphas = linspace(0,.8,tplot+2)';
for i = 1:nfids,
%  htrx(i) = patch(nan(102,1),nan(102,1),colors(i,:),...
%    'EdgeAlpha','interp','EdgeColor',colors(i,:)*.7+.3,'FaceColor','none',...
%    'FaceVertexAlphaData',alphas,'LineWidth',2);   
   htrx(i) = plot(nan,nan,'-','Color',colors(i,:)*.5+.5,'LineWidth',2);
end
for i = 1:nfids,
  hcurr(i) = plot(nan,nan,'.-','LineWidth',3,'Color',colors(i,:)*.7,'MarkerSize',10);
%   hcurr(i) = patch(nan,nan,colors(i,:),...
%     'EdgeAlpha',1,'FaceColor','none','LineWidth',3,'EdgeColor',colors(i,:)*.7);
  %hcurr(i) = text(nan,nan,num2str(i),'Color',colors(i,:),'HorizontalAlignment','center','VerticalAlignment','middle');
end
htext = text(.5,.5,'0 s','HorizontalAlignment','left','VerticalAlignment','top','Color',[.8,0,.8],'FontSize',36);
hax = gca;
set(hax,'FontSize',24,'Color','k');

p = phisPr2{fly};

fil = normpdf(-6:6,0,2);
fil = fil / sum(fil);
p = imfilter(p,fil','same','replicate','corr');

ax = [.5,.5,size(im,2)+.5,size(im,1)+.5];
border = 20;
axis off;
truesize;

[~,n] = fileparts(expdir);
% vidobj = VideoWriter(sprintf('TrackingResults_%s.avi',savestr));
% open(vidobj);

gfdata = [];
for t = 1:endframe-firstframe+1,
  
  [im,timestamp] = readframe(firstframe+t-1);
  imsz = size(im);
  set(him,'CData',im);
  for i = 1:nfids,
    
%     set(htrx(i),'XData',[p(max(1,t-tplot):t,i);nan],...
%         'YData',[p(max(1,t-tplot):t,i+nfids);nan],...
%         'FaceVertexAlphaData',alphas(max(1,tplot+2-t):end));
    set(htrx(i),'XData',p(t:-1:max(1,t-tplot),i),'YData',p(t:-1:max(1,t-tplot),i+nfids));
%    set(hcurr(i),'XData',p(t,i),'YData',p(t,nfids+i));
    if mod(i,2) == 1,
      if i < nfids,
        set(hcurr(i),'XData',p(t,[i,i+1]),'YData',p(t,[nfids+i,nfids+i+1]));
      else
        set(hcurr(i),'XData',p(t,[i,1]),'YData',p(t,[nfids+i,nfids+1]));
      end
    end
    %set(hcurr(i),'Position',[p(t,i),p(t,nfids+i),0]);
  end

%   offt = firstframe+t-1-trx(fly).firstframe+1;
%   xcurr = [trx(fly).bbs(offt,1),trx(fly).bbs(offt,1)+trx(fly).bbs(offt,3)];
%   ycurr = [trx(fly).bbs(offt,2),trx(fly).bbs(offt,2)+trx(fly).bbs(offt,4)];
%   
%   if any(xcurr - border < ax(1) | ...
%       xcurr + border > ax(2) | ...
%       ycurr - border < ax(3) | ...
%       ycurr + border > ax(4)),
%     
%     ax = [xcurr(2)-2*winrad,xcurr(2)+2*winrad,ycurr(2)-2*winrad,ycurr(2)+2*winrad];
%     set(hax,'XLim',ax(1:2),'YLim',ax(3:4));
%     set(htext,'Position',[ax(1)+5,ax(3)+5,0]);
%   end
  set(htext,'String',sprintf('%.3fs',t/fps));
  
  %pause(.25);
  drawnow;
  
%   if isempty(gfdata),
%     gfdata = getframe_initialize(hax);
%     fr = getframe_invisible(hax);
%     gfdata.sz = size(fr);
%   end
%   fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
%   writeVideo(vidobj,fr);
  
end
% close(vidobj);

if fid > 0,
  fclose(fid);
end
