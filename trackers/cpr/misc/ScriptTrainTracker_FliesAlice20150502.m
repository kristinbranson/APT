% try a variety of parameters

%clear all
doeq = false;
loadH0 = false;
cpr_type = 2;
docacheims = false;
maxNTr = 10000;
radius = 100;
winrad = 50;

addpath ..;
addpath ../video_tracking;
addpath c:\data.not.os\bitbucket\jctrax\misc
addpath c:\data.not.os\bitbucket\jctrax\filehandling;
addpath(genpath('c:\data.not.os\bitbucket\piotr_toolbox'));

defaultfolder = 'f:\cpr\data';
%defaultfile = 'M134_M174_20150423.mat';
defaultfile = 'FlyBubbleTestData_20150502.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  fullfile(defaultfolder,defaultfile));
if isnumeric(file),
  return;
end

[~,savestr] = fileparts(file);

ld = load(fullfile(folder,file));

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
  imagesc(im,'Parent',hax(ii),[0,255]);
  axis(hax(ii),'image','off');
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
    H=nan(256,nTr);
    mu = nan(1,nTr);
    for i=1:nTr,
      H(:,i)=imhist(IsTr{idx(i)});
      mu(i) = mean(IsTr{idx(i)}(:));
    end
    % normalize to brighter movies, not to dimmer movies
    idxuse = mu >= prctile(mu,75);
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
  imagesc(im,'Parent',hax(ii),[0,255]);
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
allPhisTr = reshape(permute(pts_norm,[3,1,2]),[size(pts_norm,3),size(pts_norm,1)*size(pts_norm,2)]);
tmp2 = struct;
tmp2.phisTr = allPhisTr(idx,:);
tmp2.bboxesTr = repmat([1,1,winrad,winrad],[numel(idx),1]);
tmp2.IsTr = IsTr(idx);
save(sprintf('f:\\cpr\\data\\TrainData_%s.mat',savestr),'-struct','tmp2');

%% train tracker

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyBubble1';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
params.ftr_type = 5;
params.ftr_gen_radius = winrad;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('f:\\cpr\\data\\',sprintf('TrainData_%s.mat',savestr));
paramsfile2 = sprintf('f:\\cpr\\data\\TrainParams_%s.mat',savestr);
trainresfile = sprintf('f:\\cpr\\data\\TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

npts = size(pts_norm,1);
X = reshape(pts_norm,[size(pts_norm,1)*size(pts_norm,2),size(pts_norm,3)]);
[idxinit,initlocs] = mykmeans(X',params.prunePrm.numInit,'Start','furthestfirst','Maxiter',100);
tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
if doeq,
  tmp.H0 = H0;
end
tmp.prunePrm.motion_2dto3D = false;
tmp.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-append','-struct','tmp');

%% test tracker on the labeled fly

expdir = ld.expdirs{1};
fly = ld.flies(1);
firstframe = 1;
endframe = 1000;
testresfile = sprintf('f:\\cpr\\data\\TestResults_%s',savestr);
%parfor fly = fliesAL %3:9,
  [phisPr(fly),phisPrAll(fly)]=test(expdir,trainresfile,testresfile,'moviefilestr',ld.moviefilestr,...
  'trxfilestr',ld.trxfilestr,'winrad',winrad,'flies',fly,...
  'firstframe',firstframe,'endframe',endframe);
%end

%% visualize cpr

fly = 1;
%%AL
t = 40077;
params.cascade_depth=100;
%%$\AL
[tmp1,tmp2,pt] = test(expdir,trainresfile,testresfile,'moviefilestr',ld.moviefilestr,...
  'trxfilestr',ld.trxfilestr,'winrad',winrad,'flies',fly,...
  'firstframe',t,'endframe',t+4);

ptcurr = reshape(pt{1}(1,:,:,:),[params.prunePrm.numInit,params.model_nfids,params.model_d,params.cascade_depth+1]);

load(fullfile(expdir,ld.trxfilestr));
[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));

im = readframe(t);
toff = t-trx(fly).firstframe+1;
xcurr = round(trx(fly).x(toff));
ycurr = round(trx(fly).y(toff));
ax = [xcurr-winrad,xcurr+winrad,ycurr-winrad,ycurr+winrad];

clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
set(gca,'XTick',[],'YTick',[]);

if ~exist('CPRIterations','dir'),
  mkdir('CPRIterations');
end

SaveFigLotsOfWays(gcf,'CPRIterations/RawImage');

hold on;
r = 4;
i = 1;
for j = 1:npts,
  plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',12,'LineWidth',2);
end

SaveFigLotsOfWays(gcf,'CPRIterations/Initialization');

clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
hax = gca;
set(gca,'XTick',[],'YTick',[]);

i = params.cascade_depth+1;
hold on;
for j = 1:npts,
  plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',12,'LineWidth',2);
end

SaveFigLotsOfWays(gcf,'CPRIterations/Final');


clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
hax = gca;
set(gca,'XTick',[],'YTick',[]);

i = params.cascade_depth+1;
k = 2;
hold on;
for j = 1:npts,
  plot(ptcurr(r,j,1,k),ptcurr(r,j,2,k),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',12,'LineWidth',2);
  h = quiver(ptcurr(r,j,1,k),ptcurr(r,j,2,k),ptcurr(r,j,1,i)-ptcurr(r,j,1,k),ptcurr(r,j,2,i)-ptcurr(r,j,2,k),0);
  set(h,'LineWidth',3,'Color',colors(j,:),'MaxHeadSize',1);
end

SaveFigLotsOfWays(gcf,sprintf('CPRIterations/Update%d',k));


clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
set(gca,'XTick',[],'YTick',[]);

hold on;
i = 3;
for j = 1:npts,
  plot(squeeze(ptcurr(r,j,1,1:i-1)),squeeze(ptcurr(r,j,2,1:i-1)),'--','Color',colors(j,:)*.7,'MarkerSize',12,'LineWidth',2);
  plot(squeeze(ptcurr(r,j,1,i-1:i)),squeeze(ptcurr(r,j,2,i-1:i)),'x-','Color',colors(j,:)*.7,'MarkerSize',12,'LineWidth',3);
  plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',12,'LineWidth',2);
end

SaveFigLotsOfWays(gcf,sprintf('CPRIterations/Iteration%02d',i));


vidobj = VideoWriter(sprintf('CPRIterations/Iterations.avi'));
vidobj.FrameRate = 10;
open(vidobj);

gfdata = [];
figure(1);
colormap gray;
for i = 1:params.cascade_depth,
  clf;
  hfig = gcf;
  imagesc(im,[0,255]);
  axis image;
  axis(ax);
  hax = gca;
  set(gca,'XTick',[],'YTick',[]);
  
  hold on;
  for j = 1:npts,
    plot(squeeze(ptcurr(r,j,1,1:i)),squeeze(ptcurr(r,j,2,1:i)),'o-','Color',colors(j,:)*.7,'MarkerSize',8,'LineWidth',5,'MarkerFaceColor',colors(j,:)*.7);
  end
  for j = 1:npts,
    plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',20,'LineWidth',4);
  end
  text(ax(1),ax(3),sprintf('  Iter %d',i),'FontSize',36,'HorizontalAlignment','left','VerticalAlignment','top');
  drawnow;
  if isempty(gfdata),
    gfdata = getframe_initialize(hax);
    fr = getframe_invisible(hax);
    gfdata.sz = size(fr);
  end
  gfdata.hfig = hfig;
  gfdata.haxes = gca;
  gfdata.hardcopy_args{1} = gca;
  fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
  writeVideo(vidobj,fr);
  
end
close(vidobj);

MakeTrackingResultsHistogramVideo(expdir,testresfile,'moviefilestr','movie.ufmf','intrxfile',fullfile(expdir,ld.trxfilestr),'fly',1,'winrad',winrad,'TrxColor','k','TextColor','k')


vidobj = VideoWriter(sprintf('CPRIterations/IterationsRestarts.avi'));
vidobj.FrameRate = 10;
open(vidobj);

gfdata = [];

hfig = 3;
figure(hfig);
for i = 1:params.cascade_depth+1,

clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
set(gca,'XTick',[],'YTick',[]);
hax = gca;

hold on;
%i = params.cascade_depth+1;
smoothsig = 2;
binedges{1} = floor(ax(1)):ceil(ax(2));
binedges{2} = floor(ax(3)):ceil(ax(4));
bincenters{1} = (binedges{1}(1:end-1)+binedges{1}(2:end))/2;
bincenters{2} = (binedges{2}(1:end-1)+binedges{2}(2:end))/2;
counts = cell(1,npts);
fil = fspecial('gaussian',6*smoothsig+1,smoothsig);
maxv = .15;
for j = 1:npts,
  counts{j} = hist3([ptcurr(:,j,1,i),ptcurr(:,j,2,i)],'edges',binedges);
  counts{j} = counts{j}(1:end-1,1:end-1) / params.prunePrm.numInit;
  counts{j} = imfilter(counts{j},fil,'corr','same',0);
  him2 = image([bincenters{1}(1),bincenters{1}(end)],[bincenters{2}(1),bincenters{2}(end)],...
    repmat(reshape(colors(j,:),[1,1,3]),size(counts{j}')),...
    'AlphaData',min(1,3*sqrt(counts{j}')/sqrt(maxv)),'AlphaDataMapping','none');
end

for r = 1:params.prunePrm.numInit,
  for j = 1:npts,
    %plot(squeeze(ptcurr(r,j,1,1:i)),squeeze(ptcurr(r,j,2,1:i)),'-','Color',colors(j,:)*.7,'LineWidth',1);
    if ptcurr(r,j,1,i) < ax(1) || ptcurr(r,j,1,i) > ax(2) || ...
        ptcurr(r,j,2,i) < ax(3) || ptcurr(r,j,2,i) > ax(4),
      continue;
    end
    plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'o','Color',colors(j,:),'MarkerFaceColor',colors(j,:),'MarkerSize',6,'LineWidth',1);
  end
end

text(ax(1),ax(3),sprintf('  Iter %d',i),'FontSize',36,'HorizontalAlignment','left','VerticalAlignment','top');


drawnow;

  if isempty(gfdata),
    gfdata = getframe_initialize(hax);
    fr = getframe_invisible(hax);
    gfdata.sz = size(fr);
  end
  gfdata.hfig = hfig;
  gfdata.haxes = gca;
  gfdata.hardcopy_args{1} = gca;
  fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
  writeVideo(vidobj,fr);
  
end
close(vidobj);


SaveFigLotsOfWays(gcf,sprintf('CPRIterations/Restarts'));

clf;
imagesc(im,[0,255]);
axis image;
axis(ax);
set(gca,'XTick',[],'YTick',[]);

hold on;
r = 5;
i = 1;
for j = 1:npts,
  plot(ptcurr(r,j,1,i),ptcurr(r,j,2,i),'wo','MarkerFaceColor',colors(j,:),'MarkerSize',12,'LineWidth',2);
end
SaveFigLotsOfWays(gcf,'CPRIterations/Initialization2');

%%

fly = 1;

nfids = size(phisPr{1},2)/2;

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
trx = load_tracks(fullfile(expdir,ld.trxfilestr));

fprintf('Initial tracking results\n');

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrx = nan(1,nfids);
hcurr = nan(1,nfids);
colors = jet(nfids);
alphas = linspace(0,1,102)';
for i = 1:nfids,
  htrx(i) = patch(nan(102,1),nan(102,1),colors(i,:),...
    'EdgeAlpha','interp','EdgeColor',colors(i,:)*.7,'FaceColor','none',...
    'FaceVertexAlphaData',alphas,'LineWidth',5);   
  %htrx(i) = plot(nan,nan,'.-','Color',colors(i,:)*.7);
  hcurr(i) = plot(nan,nan,'o','LineWidth',2,'Color',colors(i,:),'MarkerSize',10);
end
htext = text(nan,nan,'0 s','HorizontalAlignment','left','VerticalAlignment','top','Color',[.8,0,.8],'FontSize',36);
hax = gca;
set(hax,'FontSize',24,'Color','k');

p = phisPr{fly};

ax = [0,0,0,0];
border = 20;

box off;

[~,n] = fileparts(expdir);
vidobj = VideoWriter(sprintf('TrackingResults_%s_Fly%d_20150502.avi',n,fly));
open(vidobj);

gfdata = [];
for t = 1:endframe-firstframe+1,
  
  [im,timestamp] = readframe(firstframe+t-1);
  imsz = size(im);
  set(him,'CData',im);
  for i = 1:nfids,
    
    set(htrx(i),'XData',[p(max(1,t-100):t,i);nan],...
        'YData',[p(max(1,t-100):t,i+nfids);nan],...
        'FaceVertexAlphaData',alphas(max(1,102-t):end));
    %set(htrx(i),'XData',p(max(1,t-500):t,i),'YData',p(max(1,t-500):t,i+nfids));
    set(hcurr(i),'XData',p(t,i),'YData',p(t,nfids+i));
  end

  offt = firstframe+t-1-trx(fly).firstframe+1;
  xcurr = trx(fly).x(offt) + cos(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
  ycurr = trx(fly).y(offt) + sin(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
  
  if any(xcurr - border < ax(1) | ...
      xcurr + border > ax(2) | ...
      ycurr - border < ax(3) | ...
      ycurr + border > ax(4)),
    
    ax = [xcurr(2)-2*winrad,xcurr(2)+2*winrad,ycurr(2)-2*winrad,ycurr(2)+2*winrad];
    set(hax,'XLim',ax(1:2),'YLim',ax(3:4));
    set(htext,'Position',[ax(1)+5,ax(3)+5,0]);
  end
  set(htext,'String',sprintf('%.3f',timestamp));
  
  %pause(.25);
  drawnow;
  
  if isempty(gfdata),
    gfdata = getframe_initialize(hax);
    fr = getframe_invisible(hax);
    gfdata.sz = size(fr);
  end
  fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
  writeVideo(vidobj,fr);
  
end
close(vidobj);

fclose(fid);

%%

nflies = numel(phisPr);
nfids = size(phisPr{1},2)/2;

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
trx = load_tracks(fullfile(expdir,ld.trxfilestr));

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrx = nan(nflies,nfids);
hcurr = nan(nflies,nfids);
colors = jet(nfids);
alphas = linspace(0,.5,102)';
for fly = 1:nflies,
  for i = 1:nfids,
    htrx(fly,i) = patch(nan(102,1),nan(102,1),colors(i,:),...
      'EdgeAlpha','interp','EdgeColor',colors(i,:)*.7,'FaceColor','none',...
      'FaceVertexAlphaData',alphas,'LineWidth',2);
    %htrx(fly,i) = plot(nan,nan,'-','Color',colors(i,:)*.7);
    hcurr(fly,i) = plot(nan,nan,'.','Color',colors(i,:),'MarkerSize',16);
  end
end
htext = text(5.5,5.5,'0 s','HorizontalAlignment','left','VerticalAlignment','top','Color',[.8,0,.8],'FontSize',36);
hax = gca;
set(hax,'FontSize',24,'Color','k');
truesize;
axis off;
%ax = [0,0,0,0];
%border = 20;

box off;

[~,n] = fileparts(expdir);
vidobj = VideoWriter(sprintf('TrackingResults_%s_AllFlies_20150502.avi',n));
open(vidobj);
gfdata = [];

for t = 1:endframe-firstframe+1,
  
  [im,timestamp] = readframe(firstframe+t-1);
  imsz = size(im);
  set(him,'CData',im);
  for fly = 1:nflies,
    for i = 1:nfids,
      set(htrx(fly,i),'XData',[phisPr{fly}(max(1,t-100):t,i);nan],...
        'YData',[phisPr{fly}(max(1,t-100):t,i+nfids);nan],...
        'FaceVertexAlphaData',alphas(max(1,102-t):end));
      set(hcurr(fly,i),'XData',phisPr{fly}(t,i),'YData',phisPr{fly}(t,nfids+i));
    end
  end

  offt = firstframe+t-1-trx(fly).firstframe+1;
  xcurr = trx(fly).x(offt) + cos(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
  ycurr = trx(fly).y(offt) + sin(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
  
%   if any(xcurr - border < ax(1) | ...
%       xcurr + border > ax(2) | ...
%       ycurr - border < ax(3) | ...
%       ycurr + border > ax(4)),
%     
%     ax = [xcurr(2)-2*winrad,xcurr(2)+2*winrad,ycurr(2)-2*winrad,ycurr(2)+2*winrad];
%     set(hax,'XLim',ax(1:2),'YLim',ax(3:4));
%     set(htext,'Position',[ax(1)+5,ax(3)+5,0]);
%   end
  set(htext,'String',sprintf('%.3f',timestamp));
  
  %pause(.25);
  drawnow;
  if isempty(gfdata),
    gfdata = getframe_initialize(hax);
    fr = getframe_invisible(hax);
    gfdata.sz = size(fr);
  end
  fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
  writeVideo(vidobj,fr);
  
end
close(vidobj);

fclose(fid);


%% Train model for point 1

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

paramsfile1_pt1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt1.mat',savestr));
paramsfile2_pt1 = sprintf('TrainParams_%s_2D_pt1.mat',savestr);
trainresfile_pt1 = sprintf('TrainModel_%s_2D_pt1.mat',savestr);

save(paramsfile2_pt1,'-struct','params');

allPhisTr2 = allPhisTr(:,[1 3]);
phisTr2 = allPhisTr2(idx,:);
bboxesTr2 = [phisTr2-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr2))];
copyfile(paramsfile1,paramsfile1_pt1);
tmp3 = struct;
tmp3.phisTr = phisTr2;
tmp3.bboxesTr = bboxesTr2;
%tmp3.IsTr = IsTr(idx);
save(paramsfile1_pt1,'-append','-struct','tmp3');

[regModel_pt1,regPrm_pt1,prunePrm_pt1,phisPr_pt1,err_pt1] = train(paramsfile1_pt1,paramsfile2_pt1,trainresfile_pt1);

tmp = load(trainresfile_pt1);
tmp.H0 = H0;
save(trainresfile_pt1,'-struct','tmp');

%% Train model for point 2

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,medfilwidth,params.prunePrm.windowradius);

paramsfile1_pt2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt2.mat',savestr));
paramsfile2_pt2 = sprintf('TrainParams_%s_2D_pt2.mat',savestr);
trainresfile_pt2 = sprintf('TrainModel_%s_2D_pt2.mat',savestr);

save(paramsfile2_pt2,'-struct','params');

allPhisTr3 = allPhisTr(:,[2 4]);
phisTr3 = allPhisTr3(idx,:);
bboxesTr3 = [phisTr3-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr3))];

copyfile(paramsfile1,paramsfile1_pt2);
tmp4 = struct;
tmp4.phisTr = phisTr3;
tmp4.bboxesTr = bboxesTr3;
%tmp4.IsTr = IsTr(idx);
save(paramsfile1_pt2,'-append','-struct','tmp4');
%save(paramsfile1_pt2,'-struct','tmp4');

[regModel_pt2,regPrm_pt2,prunePrm_pt2,phisPr_pt2,err_pt2] = train(paramsfile1_pt2,paramsfile2_pt2,trainresfile_pt2);

save(trainresfile_pt2,'-append','H0');

allregressors = struct;
allregressors.regModel = cell(1,3);
allregressors.regPrm = cell(1,3);
allregressors.prunePrm = cell(1,3);
allregressors.H0 = H0;
allregressors.traindeps = [0,1,1];

tmp = load(trainresfile);
allregressors.regModel{1} = tmp.regModel;
allregressors.regPrm{1} = tmp.regPrm;
allregressors.prunePrm{1} = tmp.prunePrm;
tmp = load(trainresfile_pt1);
allregressors.regModel{2} = tmp.regModel;
allregressors.regPrm{2} = tmp.regPrm;
allregressors.prunePrm{2} = tmp.prunePrm;
tmp = load(trainresfile_pt2);
allregressors.regModel{3} = tmp.regModel;
allregressors.regPrm{3} = tmp.regPrm;
allregressors.prunePrm{3} = tmp.prunePrm;

trainresfile_combine = sprintf('TrainedModel_%s_2D_combined.mat',savestr);
save(trainresfile_combine,'-struct','allregressors');

%% version 2 with motion-based prediction


allregressors = struct;
allregressors.regModel = cell(1,3);
allregressors.regPrm = cell(1,3);
allregressors.prunePrm = cell(1,3);
allregressors.H0 = H0;
allregressors.traindeps = [0,1,1];

tmp = load(trainresfile);
allregressors.regModel{1} = tmp.regModel;
allregressors.regPrm{1} = tmp.regPrm;
allregressors.prunePrm{1} = tmp.prunePrm;
%allregressors.prunePrm{1}.motion_2dto3D = true;
%allregressors.prunePrm{1}.calibrationdata = calibrationdata;
% turned this off 20150427 because points in each view are different
allregressors.prunePrm{1}.motion_2dto3D = false;
allregressors.prunePrm{1}.motionparams = {'poslambda',.5};

tmp = load(trainresfile_pt1);
allregressors.regModel{2} = tmp.regModel;
allregressors.regPrm{2} = tmp.regPrm;
allregressors.prunePrm{2} = tmp.prunePrm;
allregressors.prunePrm{2}.motion_2dto3D = false;
allregressors.prunePrm{2}.motionparams = {'poslambda',.75};
allregressors.prunePrm{2}.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,0,tmp.prunePrm.windowradius);

tmp = load(trainresfile_pt2);
allregressors.regModel{3} = tmp.regModel;
allregressors.regPrm{3} = tmp.regPrm;
allregressors.prunePrm{3} = tmp.prunePrm;
allregressors.prunePrm{3}.motion_2dto3D = false;
allregressors.prunePrm{3}.motionparams = {'poslambda',.75};
allregressors.prunePrm{3}.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,0,tmp.prunePrm.windowradius);

trainresfile_motion_combine = sprintf('TrainedModel_%s_2D_motion_combined.mat',savestr);
save(trainresfile_motion_combine,'-struct','allregressors');

%% track movie

firstframe = 51;
endframe = 250;
expdir = '/tier2/hantman/Jay/videos/M174VGATXChR2/20150416L/L2secOn3Grab/M174_20150416_v007';
testresfile = '';
[phisPr,phisPrAll]=test(expdir,trainresfile_motion_combine,testresfile,ld.moviefilestr,[],firstframe,endframe);

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
imsz = size(im);

fprintf('Initial tracking results\n');

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
hother = nan(1,2);
hother(1) = plot(nan,nan,'b.');
hother(2) = plot(nan,nan,'b.');
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m.-');
htrx(2) = plot(nan,nan,'c.-');
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

idxcurr = idx(ld.expidx(idx)==expi);
tstraincurr = ld.ts(idxcurr);

%p = phisPr{1};
p1 = phisPr{2};
p2 = phisPr{3};

for t = 1:endframe-firstframe+1,
  
  im = readframe(firstframe+t-1);
  set(him,'CData',im);
%   set(hother(1),'XData',squeeze(pall(t,1,:)),'YData',squeeze(pall(t,3,:)));
%   set(hother(2),'XData',squeeze(pall(t,2,:)),'YData',squeeze(pall(t,4,:)));
  set(htrx(1),'XData',p1(max(1,t-500):t,1),'YData',p1(max(1,t-500):t,2));
  set(htrx(2),'XData',p2(max(1,t-500):t,1),'YData',p2(max(1,t-500):t,2));
  set(hcurr(1),'XData',p1(t,1),'YData',p1(t,2));
  set(hcurr(2),'XData',p2(t,1),'YData',p2(t,2));
  %pause(.25);
  drawnow;
  
end

%% track more movies

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

rootdirs = {%'/tier2/hantman/Jay/videos/M118_CNO_G6'
  %'/tier2/hantman/Jay/videos/M119_CNO_G6'
  %'/tier2/hantman/Jay/videos/M122_CNO_M1BPN_G6_anno'
  %'/tier2/hantman/Jay/videos/M127D22_hm4D_Sl1_BPN'
  %'/tier2/hantman/Jay/videos/M130_hm4DBPN_KordM1'
  %'/tier2/hantman/Jay/videos/M134C3VGATXChR2'
  '/tier2/hantman/Jay/videos/M174VGATXChR2'
  %'/tier2/hantman/Jay/videos/M147VGATXChrR2_anno'
  };

expdirs = cell(1,numel(rootdirs));
exptypes = cell(1,numel(rootdirs));

for i = 1:numel(rootdirs),
  
  [~,exptypes{i}] = fileparts(rootdirs{i});
  expdirs{i} = recursiveDir(rootdirs{i},'movie_comb.avi');
  
  expdirs{i} = cellfun(@(x) fileparts(x),expdirs{i},'Uni',0);
  
end

saverootdir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150430';
if ~exist(saverootdir,'dir'),
  mkdir(saverootdir);
end

scriptfiles = {};
outfiles = {};
testresfiles = {};

for typei = numel(rootdirs):-1:1,
  
  savedircurr = fullfile(saverootdir,exptypes{typei});
  if ~exist(savedircurr,'dir'),
    mkdir(savedircurr);
  end
    
  for i = 1:numel(expdirs{typei}),
    
    expdir = expdirs{typei}{i};
    [~,n] = fileparts(expdir);
    
    
    if numel(testresfiles)>=typei && numel(testresfiles{typei}) >= i && ...
        ~isempty(testresfiles{typei}{i}) && exist(testresfiles{typei}{i},'file'),
      continue;
    end
    
    if numel(scriptfiles) >= typei && numel(scriptfiles{typei}) >= i && ~isempty(scriptfiles{typei}{i}),
      [~,jobid] = fileparts(scriptfiles{typei}{i});
    else
      jobid = sprintf('track_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
      testresfiles{typei}{i} = fullfile(savedircurr,[jobid,'.mat']);
      scriptfiles{typei}{i} = fullfile(savedircurr,[jobid,'.sh']);
      outfiles{typei}{i} = fullfile(savedircurr,[jobid,'.log']);
    end

    fid = fopen(scriptfiles{typei}{i},'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %s %s %s\n',...
      SCRIPT,MCR,expdir,trainresfile_motion_combine,testresfiles{typei}{i},ld.moviefilestr);
    fclose(fid);
    unix(sprintf('chmod u+x %s',scriptfiles{typei}{i}));
  
    cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
      curdir,NCORESPERJOB,jobid,outfiles{typei}{i},scriptfiles{typei}{i});

    unix(cmd);
  end
  
end


% load and combine files

for typei = numel(rootdirs):-1:1,
  
  parentdirs = regexp(expdirs{typei},['^',rootdirs{typei},'/([^/]*)/'],'tokens','once');
  parentdirs = [parentdirs{:}];
  [uniqueparentdirs,~,parentidx] = unique(parentdirs);
  
  for pdi = 1:numel(uniqueparentdirs),
    
    savefile = fullfile(saverootdir,sprintf('TrackingResults_%s_%s_%s.mat',exptypes{typei},uniqueparentdirs{pdi},datestr(now,'yyyymmdd')));
    if exist(savefile,'file'),
      continue;
    end
    savestuff = struct;
    savestuff.curr_vid = 1;
    savestuff.moviefiles_all = {};
    savestuff.p_all = {};
    savestuff.hyp_all = {};
  
    idxcurr = find(parentidx==pdi);
    for ii = 1:numel(idxcurr),
      
      i = idxcurr(ii);
      expdir = expdirs{typei}{i};
      if ~exist(testresfiles{typei}{i},'file'),
        error('%s does not exist.\n',testresfiles{typei}{i});
      end
      savestuff.moviefiles_all{1,ii} = fullfile(expdir,ld.moviefilestr);
      tmp = load(testresfiles{typei}{i});
      savestuff.p_all{ii,1}(:,[1,3]) = tmp.phisPr{2};
      savestuff.p_all{ii,1}(:,[2,4]) = tmp.phisPr{3};
      savestuff.hyp_all{ii,1}(:,[1,3],:) = tmp.phisPrAll{2};
      savestuff.hyp_all{ii,1}(:,[2,4],:) = tmp.phisPrAll{3};
    end
    
    fprintf('Saving tracking results for %d videos to %s\n',numel(idxcurr),savefile);
    save(savefile,'-struct','savestuff');
  end
    
end

%% plot tracking results!

typei = 1;
order = randsample(numel(testresfiles{typei}),20);

vidobj = VideoWriter('TrackingResults_RandomVideos_20150430.avi');
open(vidobj);
gfdata = [];

for i = order(3:end)',
  
  expdir = expdirs{typei}{i};
  trxfile = testresfiles{typei}{i};
  
  load(trxfile,'phisPr'),
  d0 = sqrt(sum(bsxfun(@minus,phisPr{end},phisPr{end}(1,:)).^2,2)+...
    sum(bsxfun(@minus,phisPr{end-1},phisPr{end-1}(1,:)).^2,2));
  d1 = sqrt(sum(bsxfun(@minus,phisPr{end},phisPr{end}(end,:)).^2,2)+...
    sum(bsxfun(@minus,phisPr{end-1},phisPr{end-1}(end,:)).^2,2));
  firstframe = max(1,find(d0 > 20,1)-25);
  endframe = min(size(phisPr{end},1),find(d1 > 20,1,'last')+25);
    
  [vidobj] = MakeTrackingResultsHistogramVideo(expdir,trxfile,'firstframe',firstframe,'endframe',endframe,...
    'vidobj',vidobj);
  
end

close(vidobj);

%% train first tracker with cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw2';
params.ftr_type = 6;
params.ftr_gen_radius = 100;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr));
paramsfile2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainParams_%s_2D_20150427_CV.mat',savestr));
trainresfile = sprintf('TrainedModel_%s_CV.mat',savestr);

params.expidx = ld.expidx(idx);
params.cvidx = CVSet(params.expidx,params.ncrossvalsets);

save(paramsfile2,'-struct','params');

trainsavedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/CrossValidationResults20150430';
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end

NCORESPERJOB = 4;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

trainresfiles = cell(1,params.ncrossvalsets);
trainscriptfiles = cell(1,params.ncrossvalsets);
trainoutfiles = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles{cvi},trainscriptfiles{cvi});

  unix(cmd);
end

% collect the results
td = struct('regModel',{{}},'regPrm',[],'prunePrm',[],...
  'phisPr',nan(numel(params.cvidx),4),'paramfile1','','err',nan(1,params.ncrossvalsets),...
  'paramfile2','','cvidx',cvidx);
for cvi = 1:params.ncrossvalsets,
 
  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
    continue;
  end
  tdcurr = load(trainresfiles{cvi});
  td.regModel{cvi} = tdcurr.regModel;
  if cvi == 1,
    td.regPrm = tdcurr.regPrm;
    td.prunePrm = tdcurr.prunePrm;
    td.paramfile1 = tdcurr.paramfile1;
    td.paramfile2 = tdcurr.paramfile2;
    td.cvidx = tdcurr.cvidx;
  end
  if isfield(td,'phisPr'),
    td.phisPr(cvidx==cvi,:) = tdcurr.phisPr;
  end
  if isfield(td,'err'),
    td.err(cvi) = tdcurr.err;
  end
end

td.H0 = H0;
td.prunePrm.motion_2dto3D = false;
td.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-struct','td');

% add motion parameters to individual tracking results
for cvi = 1:params.ncrossvalsets,
  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
    continue;
  end
  tdcurr = load(trainresfiles{cvi});
  tdcurr.H0 = H0;
  tdcurr.prunePrm.motion_2dto3D = false;
  tdcurr.prunePrm.motionparams = {'poslambda',.5};
  save(trainresfiles{cvi},'-struct','tdcurr');
end

%% compute the error by testing on full training movies

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

testresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

for expi = 1:max(params.expidx),

  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr));
  assert(numel(cvi)==1);
  
  [~,n] = fileparts(expdir);
       
  jobid = sprintf('track_cv_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
  testresfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{expi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s %s\n',...
    SCRIPT,MCR,expdir,trainresfiles{cvi},testresfiles_cv{expi},ld.moviefilestr);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{expi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{expi},scriptfiles_cv{expi});

  unix(cmd);  
  
end

% collect results
trx = struct;
for expi = 1:max(params.expidx),

  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr));
  assert(numel(cvi)==1);
  
  if ~exist(testresfiles_cv{expi},'file'),
    fprintf('File %s does not exist, skipping\n',testresfiles_cv{expi});
    continue;
  end
  
  tdcurr = load(testresfiles_cv{expi});
  
  if expi == 1,
    nphi = numel(tdcurr.phisPr);
    trx.phisPr = cell(nphi,1);
    trx.phisPrAll = cell(nphi,1);
    for i = 1:nphi,
      trx.phisPr{i} = nan(numel(params.cvidx),size(tdcurr.phisPr{i},2));
      trx.phisPrAll{i} = nan([numel(params.cvidx),size(tdcurr.phisPrAll{i},2),size(tdcurr.phisPrAll{i},3)]);
    end
  end
  
  ts = ld.ts(idx(idxcurr));
  
  assert(all(ismember([repmat(expi,[numel(ts),1]),ts(:)],[ld.expidx(:),ld.ts(:)],'rows')));
  
  for i = 1:nphi,
    trx.phisPr{i}(idxcurr,:) = tdcurr.phisPr{i}(ts,:);
    trx.phisPrAll{i}(idxcurr,:,:) = tdcurr.phisPrAll{i}(ts,:,:);
  end  
end

save(fullfile(trainsavedir,sprintf('CVPredictions_%s.mat',savestr)),'-struct','trx');

%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1,paramsfile2,trainresfile,'cvi',1);

%% train point 1 tracker w cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;
%params.cascade_depth = 1;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

params.expidx = ld.expidx(idx);
load(paramsfile2,'cvidx');
params.cvidx = cvidx;

paramsfile1_pt1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt1.mat',savestr));
paramsfile2_pt1 = sprintf('TrainParams_%s_2D_pt1_CV.mat',savestr);
trainresfile_pt1 = sprintf('TrainModel_%s_2D_pt1_CV.mat',savestr);

save(paramsfile2_pt1,'-struct','params');

trainresfiles_pt1 = cell(1,params.ncrossvalsets);
trainscriptfiles_pt1 = cell(1,params.ncrossvalsets);
trainoutfiles_pt1 = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_pt1_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles_pt1{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1_pt1,paramsfile2_pt1,trainresfiles_pt1{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles_pt1{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles_pt1{cvi},trainscriptfiles_pt1{cvi});

  unix(cmd);
end


%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1_pt1,paramsfile2_pt1,trainresfile_pt1,'cvi',1);

%% train point 2 tracker w cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;
%params.cascade_depth = 1;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

params.expidx = ld.expidx(idx);
load(paramsfile2,'cvidx');
params.cvidx = cvidx;

paramsfile1_pt2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt2.mat',savestr));
paramsfile2_pt2 = sprintf('TrainParams_%s_2D_pt2_CV.mat',savestr);
trainresfile_pt2 = sprintf('TrainModel_%s_2D_pt2_CV.mat',savestr);

save(paramsfile2_pt2,'-struct','params');

trainresfiles_pt2 = cell(1,params.ncrossvalsets);
trainscriptfiles_pt2 = cell(1,params.ncrossvalsets);
trainoutfiles_pt2 = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_pt2_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles_pt2{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1_pt2,paramsfile2_pt2,trainresfiles_pt2{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles_pt2{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles_pt2{cvi},trainscriptfiles_pt2{cvi});

  unix(cmd);
end


%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1_pt2,paramsfile2_pt2,trainresfile_pt2,'cvi',1);

%% collect all 3 tracking results

allregressors_cv = struct;
allregressors_cv.regModel = cell(1,3);
allregressors_cv.regPrm = cell(1,3);
allregressors_cv.prunePrm = cell(1,3);
allregressors_cv.H0 = H0;
allregressors_cv.traindeps = [0,1,1];

trainresfiles_motion_combine = cell(1,params.ncrossvalsets);

for cvi = 1:params.ncrossvalsets,
  trainresfiles_motion_combine{cvi} = fullfile(trainsavedir,sprintf('TrainedModel_%s_cv%02d_2D_motion_combined.mat',savestr,cvi));
  
  fprintf('%s...\n',trainresfiles_motion_combine{cvi});

  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
  else
    tmp = load(trainresfiles{cvi});
    allregressors_cv.regModel{1} = tmp.regModel;
    allregressors_cv.regPrm{1} = tmp.regPrm;
    allregressors_cv.prunePrm{1} = tmp.prunePrm;
    allregressors_cv.prunePrm{1}.motion_2dto3D = false;
    allregressors_cv.prunePrm{1}.motionparams = {'poslambda',.5};
  end

  if ~exist(trainresfiles_pt1{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles_pt1{cvi});
  else
    tmp = load(trainresfiles_pt1{cvi});
    allregressors_cv.regModel{2} = tmp.regModel;
    allregressors_cv.regPrm{2} = tmp.regPrm;
    allregressors_cv.prunePrm{2} = tmp.prunePrm;
    allregressors_cv.prunePrm{2}.motion_2dto3D = false;
    allregressors_cv.prunePrm{2}.motionparams = {'poslambda',.75};
    allregressors_cv.prunePrm{2}.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,0,tmp.prunePrm.windowradius);
  end

  if ~exist(trainresfiles_pt2{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles_pt2{cvi});
  else
    tmp = load(trainresfiles_pt2{cvi});
    allregressors_cv.regModel{3} = tmp.regModel;
    allregressors_cv.regPrm{3} = tmp.regPrm;
    allregressors_cv.prunePrm{3} = tmp.prunePrm;
    allregressors_cv.prunePrm{3}.motion_2dto3D = false;
    allregressors_cv.prunePrm{3}.motionparams = {'poslambda',.75};
    allregressors_cv.prunePrm{3}.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,0,tmp.prunePrm.windowradius);
  end
  save(trainresfiles_motion_combine{cvi},'-struct','allregressors_cv');
end

%% compute the error by testing on full training movies

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

testresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

for expi = 1:numel(ld.expdirs),

  expdir = ld.expdirs{expi};
  [~,n] = fileparts(expdir);

  if ~any(ld.expidx==expi),
    continue;
  end
  
  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    fprintf('No training examples from %s\n',expdir);    
    trainfilecurr = trainresfile_motion_combine;
  else
  
    cvi = unique(params.cvidx(idxcurr));
    assert(numel(cvi)==1);
    trainfilecurr = trainresfiles_motion_combine{cvi};
  end
  
       
  jobid = sprintf('track_cv_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
  testresfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{expi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s %s\n',...
    SCRIPT,MCR,expdir,trainfilecurr,testresfiles_cv{expi},ld.moviefilestr);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{expi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{expi},scriptfiles_cv{expi});

  unix(cmd);  
  
end

% collect results
trx = struct;
for expi = 1:max(params.expidx),

  idxcurr = find(ld.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  idxcurr1 = find(ismember(idx,idxcurr));
  
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr1));
  assert(numel(cvi)==1);
  
  if ~exist(testresfiles_cv{expi},'file'),
    fprintf('File %s does not exist, skipping\n',testresfiles_cv{expi});
    continue;
  end
  
  tdcurr = load(testresfiles_cv{expi});
  
  if expi == 1,
    nphi = numel(tdcurr.phisPr);
    trx.phisPr = cell(nphi,1);
    trx.phisPrAll = cell(nphi,1);
    for i = 1:nphi,
      trx.phisPr{i} = nan(numel(params.cvidx),size(tdcurr.phisPr{i},2));
      trx.phisPrAll{i} = nan([numel(params.cvidx),size(tdcurr.phisPrAll{i},2),size(tdcurr.phisPrAll{i},3)]);
    end
  end
  
  ts = ld.ts(idxcurr);
  
  assert(all(ismember([repmat(expi,[numel(ts),1]),ts(:)],[ld.expidx(:),ld.ts(:)],'rows')));
  
  for i = 1:nphi,
    trx.phisPr{i}(idxcurr,:) = tdcurr.phisPr{i}(ts,:);
    trx.phisPrAll{i}(idxcurr,:,:) = tdcurr.phisPrAll{i}(ts,:,:);
  end  
end

save(fullfile(trainsavedir,sprintf('CVPredictions_%s.mat',savestr)),'-struct','trx');

%% plot cv error

regi = [2,3];

mouseidxcurr = mouseidx(params.expidx);
if numel(regi) == 1,
  p1 = trx.phisPr{regi}(:,[1,3]);
  p2 = trx.phisPr{regi}(:,[2,4]);
else
  p1 = trx.phisPr{regi(1)};
  p2 = trx.phisPr{regi(2)};
end
  
err = sqrt(sum((p1-allPhisTr(:,[1,3])).^2,2) + sum((p2-allPhisTr(:,[2,4])).^2,2));
nbins = 30;
[edges,ctrs] = SelectHistEdges(nbins,[0,50],'linear');
fracpermouse = nan(nmice,nbins);
for mousei = 1:nmice,
  
  idxcurr = mouseidxcurr == mousei;
  counts = hist(err(idxcurr),ctrs);
  fracpermouse(mousei,:) = counts / sum(counts);
  
end

fractotal = hist(err,ctrs);
fractotal = fractotal / sum(fractotal);

hfig = 1;

figure(hfig);
clf;
hpermouse = plot(ctrs,fracpermouse,'-');
hold on;
h = plot(ctrs,fractotal,'k-','LineWidth',3);

box off;
xlabel('Mean squared error, both views (px)');
ylabel('Fraction of data');

ylim = get(gca,'YLim');
ylim(1) = 0;
set(gca,'YLim',ylim);

prctiles = [50,75,95,99];%,100];
humanerr = [5.736611,9.073831,11.347343,13.273021];%,19.030200];
autoerr = prctile(err,prctiles);

disp(autoerr);

dy = ylim(2) / (numel(prctiles)+1);
colors = jet(256);
colors = colors(RescaleToIndex(1:numel(prctiles)+1,256),:);
colors = flipud(colors);

for j = 1:numel(prctiles),
  plot([humanerr(j),autoerr(j)],dy*j+[0,0],'-','Color',colors(j+1,:)*.7,'LineWidth',2);
  hhuman = plot(humanerr(j),dy*j,'o','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);  
  hauto = plot(autoerr(j),dy*j,'x','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);
  text(max([humanerr(j),autoerr(j)]),dy*j,sprintf('   %dth %%ile',prctiles(j)),'HorizontalAlignment','left',...
    'Color',.7*colors(j+1,:),'FontWeight','bold');
end

legend([hhuman,hauto],'Human','Tracker');

SaveFigLotsOfWays(hfig,fullfile(trainsavedir,'CVErr'));

%% put this error in context

hfig = 5;
figure(hfig);

i = 4509;

im = IsTr{idx(i)};
clf;
imagesc(im,[0,255]);
axis image;
hold on;
colormap gray;
plot(phisTr(i,1),phisTr(i,3),'+','Color',colors(1,:),'MarkerSize',12,'LineWidth',3);
plot(phisTr(i,2),phisTr(i,4),'+','Color',colors(1,:),'MarkerSize',12,'LineWidth',3);
for j = 1:numel(prctiles),
  errcurr = autoerr(j)/sqrt(2);
  theta = linspace(-pi,pi,100);
  plot(phisTr(i,1)+errcurr*cos(theta),phisTr(i,3)+errcurr*sin(theta),'-','LineWidth',2,'Color',colors(j+1,:)*.75+.25);
  plot(phisTr(i,2)+errcurr*cos(theta),phisTr(i,4)+errcurr*sin(theta),'-','LineWidth',2,'Color',colors(j+1,:)*.75+.25);

%   errcurr = humanerr(j)/sqrt(2);
%   plot(phisTr(i,1)+errcurr*cos(theta),phisTr(i,3)+errcurr*sin(theta),'--','LineWidth',2,'Color',colors(j+1,:));
%   plot(phisTr(i,2)+errcurr*cos(theta),phisTr(i,4)+errcurr*sin(theta),'--','LineWidth',2,'Color',colors(j+1,:));

end
  
SaveFigLotsOfWays(hfig,fullfile(trainsavedir,'CVErrOnImage'));
