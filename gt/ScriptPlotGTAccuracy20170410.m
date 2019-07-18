%% paths

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;

addpath(genpath('..'));

datatype = 'roian';

switch datatype,
  case 'stephen'
    rootdatadir = '/groups/branson/bransonlab/mayank/stephenCV/results';
    lblandpredfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Stephen20170411.mat';
    figsavestr = '_Stephen20170411';
    moviesavedir = '/groups/branson/home/bransonk/tracking/code/APT/gt/TrackingResults_Stephen20170411';
    nframesplot = 1000;
  case 'jan'
    rootdatadir = '/nrs/branson/mayank/janResults';
    lblandpredfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Jan20170415.mat';
    figsavestr = '_Jan20170415';
    nframesplot = 1000;
    moviesavedir = 'TrackingResults_Jan20170415';
  case 'jay'
    lblandpredfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Jay20170416.mat';
    figsavestr = '_Jay20170416';
    nframesplot = 1000;
    moviesavedir = 'TrackingResults_Jay20170416';
  case 'roian'
    lblandpredfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Roian20170416.mat';
    figsavestr = '_Roian20170416';
    nframesplot = 1000;
    moviesavedir = 'TrackingResults_Roian20170416';

end


gtmarker = '+';
predmarkers = 'osdxp';
gtcolor = [.75,0,.75];
predcolors = [
  .25,.25,1
  0,.75,.75
  .5,.5,0
  0,.75,0
  1,.5,0
];
predfns = {'pd_locs','mrf_locs','fine_locs','final_locs','cpr_2d_locs'};
prednames = {'CNN','+ 2D pose','+ fine res.','+ 3D pose + time','CPR'};
if strcmp(datatype,'jay'),
  predfns(1) = [];
  prednames(1) = [];
  predcolors(1,:) = [];
end



npredfns = numel(predfns);
%landmarkorder = [3,1,5,2,4,3];

dosavefigs = false;

%% load data

load(lblandpredfile,'lbld','cvd');

ismissing = ~isfield(lbld,predfns);
if any(ismissing),
  warning('missing some predictions!!');
  disp(predfns(ismissing));
  predfns(ismissing) = [];
  prednames(ismissing) = [];
  predcolors(ismissing,:) = [];
  npredfns = numel(predfns);
end

nviews = size(lbld.movieFilesAll,2);
nlandmarks = size(lbld.labeledpos{1},1)/nviews;
%nlandmarks = lbld.cfg.NumLabelPoints;
nlblmovies = size(lbld.movieFilesAll,1);

landmarkcolors = lines(nlandmarks);

%% which videos are labeled and have predictions

expislabeled = cellfun(@(x) any(IsLabeled(x)),lbld.labeledpos);
allpredsfound = true(nlblmovies,1);
for p = 1:npredfns,
  predcurr = cellfun(@(x) any(IsLabeled(x)),lbld.(predfns{p}));
  allpredsfound = allpredsfound & predcurr(:);
end
allpredsfound = double(allpredsfound);
allpredsfound(~expislabeled) = nan;

%% find bad labeled data -- outside of the frame
lbld.moviesz = nan([2,nlblmovies,nviews]);
for i = find(expislabeled(:)'),
  for v = 1:nviews,
    if ~any(isnan(lbld.moviesz(:,i,v))),
      continue;
    end
    info = mmfileinfo(lbld.movieFilesAll{i,v});
    lbld.moviesz(1,i,v) = info.Video.Height;
    lbld.moviesz(2,i,v) = info.Video.Width;
  end
end

for i = find(expislabeled(:)'),
  fslabeled = find(IsLabeled(lbld.labeledpos{i}));
  isbad = false(size(fslabeled));
  for v = 1:nviews,
    off = (v-1)*nlandmarks;
    isbad = isbad | vectorize(any(lbld.labeledpos{i}(off+1:off+nlandmarks,1,fslabeled) < 1,1) ...
      | any(lbld.labeledpos{i}(off+1:off+nlandmarks,1,fslabeled) > lbld.moviesz(2,i,v),1) ...
      | any(lbld.labeledpos{i}(off+1:off+nlandmarks,2,fslabeled) < 1,1) ...
      | any(lbld.labeledpos{i}(off+1:off+nlandmarks,2,fslabeled) > lbld.moviesz(1,i,v),1))';
  end
  if any(isbad),
    mstr = StephenVideo2Identifier(lbld.movieFilesAll{i,1});
    fprintf('Found bad frames for movie %s (%d)\n',mstr,i);
    for f = fslabeled(isbad),
      for v = 1:nviews,
        off = (v-1)*nlandmarks;
        fprintf('Frame %d, view %d: minx = %f, maxx = %f, width = %d, miny = %f, maxy = %f, height = %d\n',...
          f,v,min(lbld.labeledpos{i}(off+1:off+nlandmarks,1,f)),...
          max(lbld.labeledpos{i}(off+1:off+nlandmarks,1,f)),...
          lbld.moviesz(1,i,v),...
          min(lbld.labeledpos{i}(off+1:off+nlandmarks,2,f)),...
          max(lbld.labeledpos{i}(off+1:off+nlandmarks,2,f)),...
          lbld.moviesz(2,i,v));
      end
    end
    lbld.labeledpos{i}(:,:,fslabeled(isbad)) = nan;
  end
end
expislabeled = cellfun(@(x) any(IsLabeled(x)),lbld.labeledpos);


%% to normalize different flies, for each fly, compute the average distance between the landmarks

animalids = unique(lbld.animalids(expislabeled));

% meananimaldist(i,1) is the animalid
% meananimaldist(i,2:nviews+1) is the mean dist for that view
if nlandmarks == 1,
  meananimaldist = ones(numel(animalids),nviews+1);
  meananimaldist(:,1) = animalids;
else
  
  meananimaldist = nan(numel(animalids),nviews+1);
  for i = 1:numel(animalids),
    id = animalids(i);
    idxcurr = find(lbld.animalids(:)==id & expislabeled(:));
    if isempty(idxcurr),
      continue;
    end
    meandist = zeros(1,nviews);
    ndist = 0;
    for j = idxcurr(:)',
      islabeled = IsLabeled(lbld.labeledpos{j});
      for f = find(islabeled(:)'),
        for v = 1:nviews,
          off = (v-1)*nlandmarks;
          dcurr = max(pdist(lbld.labeledpos{j}(off+1:off+nlandmarks,:,f)));
          meandist(v) = meandist(v) + dcurr;
        end
      end
      ndist = ndist + nnz(islabeled);
    end
    meandist = meandist/ndist;
    meananimaldist(i,1) = id;
    meananimaldist(i,2:nviews+1) = meandist;
  end

end
  
%% compute prediction error

err = nan([nlandmarks,nviews,npredfns,0]);
normerr = nan([nlandmarks,nviews,npredfns,0]);
movieidx = nan(1,0);
labeledframes = nan(1,0);
off = 0;
allpredsfound = nan(1,nlblmovies);
for j = 1:nlblmovies,

  % which frames are labeled
  islabeled = IsLabeled(lbld.labeledpos{j});
  if ~any(islabeled),
    continue;
  end

  % mean max distance between landmarks
  id = lbld.animalids(j);
  z = meananimaldist(meananimaldist(:,1)==id,2:end);
  assert(size(z,1) == 1);
  assert(~any(isnan(z)));
  
  % whether they are predicted for all predfns
  ispred = false(1,npredfns);
  for p = 1:npredfns,
    if isempty(lbld.(predfns{p}){j}),
      ispred(p) = false;
    else
      ispredcurr = IsLabeled(lbld.(predfns{p}){j});
      ispred(p) = all(ispredcurr(islabeled));
    end
  end
  if ~all(ispred),
    fprintf('No predictions for movie %d for the following predictors:\n',j);
    fprintf('%s ',predfns{~ispred});
    fprintf('\n');
    allpredsfound(j) = 0;
    continue;
  end
  fprintf('All predictions available for movie %d\n',j);
  allpredsfound(j) = 1;
  
  nlabeledcurr = nnz(islabeled);
  
  for viewi = 1:nviews,
    labelx = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),1,islabeled);
    labely = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),2,islabeled);
    for k = 1:npredfns,
      predfn = predfns{k};
      predx = lbld.(predfn){j}((viewi-1)*nlandmarks+(1:nlandmarks),1,islabeled);
      predy = lbld.(predfn){j}((viewi-1)*nlandmarks+(1:nlandmarks),2,islabeled);
      errcurr = sqrt( (predx-labelx).^2 + (predy-labely).^2 );
      err(:,viewi,k,off+1:off+nlabeledcurr) = errcurr;
      normerr(:,viewi,k,off+1:off+nlabeledcurr) = errcurr/z(viewi);
    end
  end
  movieidx(off+1:off+nlabeledcurr) = j;
  labeledframes(off+1:off+nlabeledcurr) = find(islabeled);
  
  off = off + nlabeledcurr;
  
end

%% compute percentiles of prediction error
switch datatype,
  case 'stephen',
    prctiles_compute = [50,75,90,95,97.5,99,99.5];
  case 'jan'
    prctiles_compute = [50,75,90,95,97.5];
  case 'jay',
    prctiles_compute = [50,75,90,95,97.5];
  case 'roian',
    prctiles_compute = [50,75,90,95,97.5];
    
end
% cpr dominated by big percentiles, don't plot all of them
maxprctileplot_perpred = inf(1,npredfns);
maxprctileplot_perpred(strcmp(predfns,'cpr_2d_locs')) = 97.5;

nprctiles = numel(prctiles_compute);
normerr_prctiles = nan([nprctiles,nlandmarks,nviews,npredfns]);
for l = 1:nlandmarks,
  for v = 1:nviews,
    for k = 1:npredfns,
      normerr_prctiles(:,l,v,k) = prctile(squeeze(normerr(l,v,k,:)),prctiles_compute);
    end
  end
end

%% plot on one selected frame

colors = jet(nprctiles);

switch datatype,
  case 'stephen',
    j = 678;
    islabeled = IsLabeled(lbld.labeledpos{j});
    %islabeled = reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]);
    f = find(islabeled,1);
    %fi = 1455;
  case 'jan',
    j = 22;
    islabeled = IsLabeled(lbld.labeledpos{j});
    %islabeled = reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]);
    f = find(islabeled,1);
    %fi = 478;
  case 'jay'
    j = 1;
    islabeled = IsLabeled(lbld.labeledpos{j});
    %islabeled = reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]);
    %f = find(islabeled,1);
    f = 491;
  case 'roian'
    j = 1;
    islabeled = IsLabeled(lbld.labeledpos{j});
    f = find(islabeled,1);
end

id = lbld.animalids(j);
idi = find(meananimaldist(:,1)==id);
assert(numel(idi)==1);
renormerr_prctiles = bsxfun(@times,normerr_prctiles,reshape(meananimaldist(idi,2:end),[1,1,nviews,1]));

% which frames are labeled

readframes = cell(1,nviews);
for viewi = 1:nviews,
  readframes{viewi} = get_readframe_fcn(lbld.movieFilesAll{j,viewi});
end

%% plot percentiles of prediction error for each type of prediction

hfig = 2;
figure(hfig);
clf;

hax = createsubplots(nviews,npredfns+1,.01);
hax = reshape(hax,[nviews,npredfns+1]);

h = nan(1,nprctiles);
for viewi = 1:nviews,
  im = readframes{viewi}(f);
  if size(im,3) == 1,
    im = repmat(im,[1,1,3]);
  end
  labelx = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),1,f);
  labely = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),2,f);

  for k = npredfns+1:-1:1,
    image(im,'Parent',hax(viewi,k));
    axis(hax(viewi,k),'image','off');
    hold(hax(viewi,k),'on');
    plot(hax(viewi,k),labelx,labely,'m+');
    if k > 1,
      for p = 1:nprctiles,
        if prctiles_compute(p) > maxprctileplot_perpred(k-1),
          continue;
        end
        for l = 1:nlandmarks,
          h(p) = drawellipse(labelx(l),labely(l),0,renormerr_prctiles(p,l,viewi,k-1),renormerr_prctiles(p,l,viewi,k-1),'Color',colors(p,:),'Parent',hax(viewi,k));
        end
      end
    end
    if k == 1,
      text(5,5,sprintf('Manual, view %d',viewi),'Color','w','Parent',hax(viewi,k),...
        'VerticalAlignment','top');
    else
      text(5,5,sprintf('%s, view %d',prednames{k-1},viewi),'Color','w','Parent',hax(viewi,k),...
        'VerticalAlignment','top');
    end
  end
end

legends = cell(1,nprctiles);
for p = 1:nprctiles,
  legends{p} = sprintf('%sth %%ile',num2str(prctiles_compute(p)));
end
hl = legend(h,legends);
set(hl,'Color','k','TextColor','w');
truesize(hfig);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['TrackingErrorOnFrame_ComparePrctiles',figsavestr],{'pdf','fig','png'});
end

%% plot median error for each prediction type

hfig = 3;
figure(hfig);
clf;
hax = createsubplots(nviews,nprctiles+1,.01);
hax = reshape(hax,[nviews,nprctiles+1]);

h = nan(1,npredfns);
for viewi = 1:nviews,
  im = readframes{viewi}(f);
  if size(im,3) == 1,
    im = repmat(im,[1,1,3]);
  end
  labelx = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),1,f);
  labely = lbld.labeledpos{j}((viewi-1)*nlandmarks+(1:nlandmarks),2,f);
  
  d = 50+max(max(renormerr_prctiles(:,:,viewi,:),[],1),[],4)';
  xlim = [1,size(im,2)];
  ylim = [1,size(im,1)];
%   xlim = [min(labelx-d),max(labelx+d)];
%   ylim = [min(labely-d),max(labely+d)];
  
  for p = 1:nprctiles+1,
    
    image(im,'Parent',hax(viewi,p));
    axis(hax(viewi,p),'image','off');
    hold(hax(viewi,p),'on');
    plot(hax(viewi,p),labelx,labely,'m+');
    
    if p > 1,
      for k = 1:npredfns,
        if prctiles_compute(p-1) > maxprctileplot_perpred(k),
          continue;
        end
        for l = 1:nlandmarks,
          h(k) = drawellipse(labelx(l),labely(l),0,renormerr_prctiles(p-1,l,viewi,k),renormerr_prctiles(p-1,l,viewi,k),'Color',predcolors(k,:),'Parent',hax(viewi,p));
        end
      end
    end
    
    if p == 1,
      text(xlim(1)+5,ylim(1)+5,sprintf('Manual, view %d',v),'Color','w','Parent',hax(viewi,p),...
        'VerticalAlignment','top','HorizontalAlignment','left');
    else
      text(xlim(1)+5,ylim(1)+5,sprintf('%sth prctile, view %d',num2str(prctiles_compute(p-1)),viewi),'Color','w','Parent',hax(viewi,p),...
        'VerticalAlignment','top','HorizontalAlignment','left');
    end
  end
  
  set(hax(viewi,:),'XLim',xlim,'YLim',ylim);
  
end

hl = legend(h,prednames);
set(hl,'Color','k','TextColor','w');
truesize(hfig);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['TrackingErrorOnFrame_ComparePreds',figsavestr],{'pdf','fig','png'});
end

%% plot error for the different prediction types against each other

if ~isfield(lbld.cfg,'LabelPointNames'),
  for l = 1:nlandmarks,
    lbld.cfg.LabelPointNames{l} = sprintf('Landmark %d',l);
  end
end
if ~isfield(lbld.cfg,'ViewNames'),
  for v = 1:nviews,
    lbld.cfg.ViewNames{v} = sprintf('View %d',v);
  end
end


hfig = 4;
figure(hfig);
clf;

colors = jet(nlandmarks*nviews)*.8;
pairs = nchoosek(1:npredfns,2);

hax = createsubplots(npredfns-1,npredfns-1,.04);
hax = reshape(hax,[npredfns-1,npredfns-1]);

h = nan(1,nviews*nlandmarks);
legends = cell(1,nviews*nlandmarks);
isused = false(size(hax));
lims = prctile(normerr(:),[.1,99.99]);
for pairi = 1:size(pairs,1),
  p1 = pairs(pairi,1);
  p2 = pairs(pairi,2);
  axi = sub2ind([npredfns-1,npredfns-1],p2-1,p1);
  isused(axi) = true;
  for l = 1:nlandmarks,
    for v = 1:nviews,
      colori = sub2ind([nviews,nlandmarks],v,l);
      h(colori) = plot(hax(axi),squeeze(normerr(l,v,p1,:)),squeeze(normerr(l,v,p2,:)),'.','Color',colors(colori,:));
      hold(hax(axi),'on');
      legends{colori} = sprintf('%s, %s',lbld.cfg.LabelPointNames{l},lbld.cfg.ViewNames{v});
    end
  end
  plot(hax(axi),lims,lims,':','Color',[.6,.6,.6],'LineWidth',2);
  if pairi == 1,
    xlabel(hax(axi),sprintf('%s, Error (normalized)',prednames{p1}));
    ylabel(hax(axi),sprintf('%s, Error (normalized)',prednames{p2}));
  else
    xlabel(hax(axi),prednames{p1});
    ylabel(hax(axi),prednames{p2});
  end
  box(hax(axi),'off');
end
axis(hax(:),'equal');
set(hax,'XLim',lims,'YLim',lims);
set(hax,'XScale','log','YScale','log');
legend(h,legends);
delete(hax(~isused));
set(hfig,'Units','pixels','Position',[10,10,1460,1460]);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['TrackingErrorVs_ComparePreds',figsavestr],{'pdf','fig','png'});
end

%% plot per-landmark frac leq curves

minfracplot = .1;

N = size(normerr,4);

minerr = inf;

% normerr is nlandmarks x nviews x npreds x N
fracleqerr = cell([nlandmarks,nviews,npredfns]);
for l = 1:nlandmarks,
  for v = 1:nviews,
    for p = 1:npredfns,
      sortederr = sort(squeeze(normerr(l,v,p,:)));
      [sortederr,nleqerr] = unique(sortederr);
      fracleqerr{l,v,p} = cat(2,nleqerr./N,sortederr);
      minerr = min(minerr,fracleqerr{l,v,p}(find(fracleqerr{l,v,p}(:,1)>=minfracplot,1),2));
    end
  end
end

hfig = 5;

figure(hfig);
clf;
hax = createsubplots(nviews,nlandmarks,[.05,.075]);
hax = reshape(hax,[nviews,nlandmarks]);

minmaxerr = inf;
for p = 1:npredfns,
  minmaxerr = min(minmaxerr,prctile(vectorize(normerr(:,:,p,:)),99.9));
end

for l = 1:nlandmarks,
  for v = 1:nviews,
    hold(hax(v,l),'on');
    for p = 1:npredfns,
      h(p) = plot(hax(v,l),fracleqerr{l,v,p}(:,2),fracleqerr{l,v,p}(:,1),'-','Color',predcolors(p,:));
    end
    if l == 1 && v == 1,
      legend(h,prednames,'Location','southeast');
    end
    xlabel(hax(v,l),'Error (normalized)');
    ylabel(hax(v,l),'Frac. smaller');
    title(hax(v,l),sprintf('%s, %s',lbld.cfg.LabelPointNames{l},lbld.cfg.ViewNames{v}));
  end
end
set(hax,'XLim',[minerr,minmaxerr],'YLim',[minfracplot,1],'XScale','log');%,'YScale','log');%

if nlandmarks > 1,
  xticks = [.01,.025,.05,.10:.10:minmaxerr];
  xticks(xticks<minerr | xticks > minmaxerr) = [];
  set(hax,'XTick',xticks);
end
% yticks = [.01:.01:.05,.1:.1:1];
% yticks(yticks<minfracplot) = [];
% set(hax,'YTick',yticks);
set(hfig,'Units','pixels','Position',[10,10,2400,880]);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['FracLessEqCurves',figsavestr],{'pdf','fig','png'});
end

%% plot max error fracleq curves

% normerr is nlandmarks x nviews x npreds x N
fracleqerr = cell([1,npredfns]);
minerr = inf;
for p = 1:npredfns,
  sortederr = sort(squeeze(max(max(normerr(:,:,p,:),[],1),[],2)));
  [sortederr,nleqerr] = unique(sortederr);
  fracleqerr{p} = cat(2,nleqerr./N,sortederr);
  minerr = min(minerr,fracleqerr{p}(find(fracleqerr{p}(:,1)>=minfracplot,1),2));
end

hfig = 6;

figure(hfig);
clf;
hax = gca;

hold(hax,'on');
for p = 1:npredfns,
  h(p) = plot(hax,fracleqerr{p}(:,2),fracleqerr{p}(:,1),'-','Color',predcolors(p,:));
end
legend(h,prednames,'Location','southeast');
xlabel(hax,'Error (normalized)');
ylabel(hax,'Frac. smaller');
title(hax,'Max over all landmarks, views');
set(hax,'XLim',[minerr,minmaxerr],'YLim',[minfracplot,1],'XScale','log');%,'YScale','log');%

if nlandmarks > 1,
  xticks = [.01,.025,.05,.10:.10:minmaxerr];
  xticks(xticks<minerr | xticks > minmaxerr) = [];
  set(hax,'XTick',xticks);
end
% yticks = [.01:.01:.05,.1:.1:1];
% yticks(yticks<minfracplot) = [];
% set(hax,'YTick',yticks);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['FracMaxErrLessEqCurves',figsavestr],{'pdf','fig','png'});
end

%% look at per-animal error

maxerroverlks = reshape(max(max(normerr,[],1),[],2),[npredfns,N]);
nlabeledperanimal = hist(lbld.animalids(movieidx),1:max(lbld.animalids(movieidx)));
normerrperanimal = nan(npredfns,numel(nlabeledperanimal));
for p = 1:npredfns,
  normerrperanimal(p,:) = accumarray(lbld.animalids(movieidx),maxerroverlks(p,:)')' ./ nlabeledperanimal;
end
islabeledanimal = nlabeledperanimal > 20;
labeledanimalids = find(islabeledanimal);
normerrperanimal(:,~islabeledanimal) = [];
nlabeledperanimal(~islabeledanimal) = [];


hfig = 7;
figure(hfig);
clf;
[~,animalorder] = sort(mean(normerrperanimal,1));
bar(normerrperanimal(:,animalorder)');
hax = gca;
set(hax,'XLim',[0,numel(labeledanimalids)+1]);
hold on;
v = 1;
ylim = get(gca,'YLim');
set(hfig,'Position',[10,10,1480,680]);
set(hax,'Units','pixels');
axpos = get(hax,'Position');
w = .8;
ylim(1) = 0;
set(hax,'YLim',ylim);
dy = diff(ylim);
xlim = get(hax,'XLim');
dx = diff(xlim);
pxperx = axpos(3)/dx;
pxpery = axpos(4)/dy;

for ii = 1:numel(labeledanimalids),
  id = labeledanimalids(animalorder(ii));
  i = find(lbld.animalids(:)==id & expislabeled(:),1);
  readframe = get_readframe_fcn(lbld.movieFilesAll{i,v});
  im = readframe(1);
  [nr,nc,ncolors] = size(im);
  if ncolors == 1,
    im = repmat(im,[1,1,3]);
  end
  imh = w*nr/nc/pxpery*pxperx;
  image(ii+w/2*[-1,1],-imh*[0,1],im);
  text(ii,-imh,num2str(id),'HorizontalAlignment','center','VerticalAlignment','top');
  drawnow;
end
set(hax,'Xtick',[],'Clipping','off');
colormap(predcolors(1:npredfns,:));
%set(gca,'XTick',1:numel(labeledanimalids),'XTickLabels',num2str(labeledanimalids(animalorder)'));
set(gca,'Box','off');
xlabel('Animal id');
ylabel('Normalized error');
legend(prednames);

colormap(predcolors);

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['ErrPerAnimal',figsavestr],{'pdf','fig','png'});
end

%% show some samples of training data

nsamplesperid = 1;
idxsample = [];
erridx2animalid = lbld.animalids(movieidx);
for i = 1:numel(animalids),
  id = animalids(i);
  idxcurr = find(erridx2animalid==id);

  if isempty(idxcurr),
    continue;
  end
  if numel(idxcurr) > nsamplesperid,
    idxcurr = idxcurr(round(linspace(1,numel(idxcurr),nsamplesperid)));
  end
  idxsample = [idxsample;idxcurr]; %#ok<AGROW>
end

naxperfig = 5;
hfigbase = 100;

hfigs = PlotExampleLabelsAndPreds(lbld,predfns,movieidx(idxsample),labeledframes(idxsample),...
  'hfigbase',hfigbase,'naxperfig',naxperfig,...
  'landmarkcolors',landmarkcolors,'figpos',[10,10,2100,1500],...
  'prednames',prednames);

if dosavefigs,
  for figi = 1:numel(hfigs),
    hfig = hfigs(figi);
    set(hfig,'Color','w','InvertHardCopy','off');
    SaveFigLotsOfWays(hfig,sprintf('SampleTrainingData_%02d_%s',figi,figsavestr),{'pdf','fig','png'});
  end
end

%% plot the worst errors

nplot = 25;
winrad = 500;
normerrcurr = normerr;
sumnormerr = reshape(sum(sum(normerr,1),2),[npredfns,N]);
sumerr = reshape(sum(sum(err,1),2),[npredfns,N]);
naxperfig = 5;

for p = 1:npredfns,

  hfigbase = 100*(p+1);
  
  idxsample = nan(1,nplot);
  sumnormerrcurr = sumnormerr(p,:);

  for ii = 1:nplot,
    [maxv,i] = max(sumnormerrcurr);
    if isnan(maxv),
      break;
    end
    idxsample(ii) = i;
    expi = movieidx(i);
    f = labeledframes(i);
    sumnormerrcurr(movieidx==expi & abs(f-labeledframes)<= 500) = nan;
    fprintf('Movie %d, frame %d, err = %f\n',expi,f,maxv);
  end
  idxsample(isnan(idxsample)) = [];
  
  
  hfigs = PlotExampleLabelsAndPreds(lbld,predfns,movieidx(idxsample),labeledframes(idxsample),...
    'hfigbase',hfigbase,'naxperfig',naxperfig,...
    'landmarkcolors',landmarkcolors,'figpos',[10,10,2100,1500],...
    'prednames',prednames,...
    'err',sumerr(:,idxsample));
  
  if dosavefigs,
    for figi = 1:numel(hfigs),
      hfig = hfigs(figi);
      set(hfig,'Color','w','InvertHardCopy','off');
      SaveFigLotsOfWays(hfig,sprintf('LargestTotalError_%s_%02d%s',predfns{p},figi,figsavestr),{'pdf','fig','png'});
    end
  end
  
end

%% make videos with trk only


switch datatype,
  case 'stephen',
    figpos = [10,10,1536,516];
  case 'jan'
    figpos = [10,10,512,512];
  case 'roian',
    figpos = [10,10,644,644];
  case 'jay',
    figpos = [10,10,704,260];
  otherwise
    error('not implemented');
end

if strcmp(datatype,'jay'),
  moviepredfns = {'mrf_locs','final_locs','fine_locs'};
elseif strcmp(datatype,'roian'),
  moviepredfns = {'pd_locs','final_locs'};
else
  moviepredfns = {'pd_locs','final_locs','fine_locs'};
end

movieidentifiers = cell(1,nlblmovies);
for i = 1:nlblmovies,
  movieidentifiers{i} = VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype);
end
trkedmovies = mydir(fullfile(moviesavedir,'*_TrkVideo.avi'));
istrkedmovie = false(1,nlblmovies);
for i = 1:numel(trkedmovies),
  [~,n] = fileparts(trkedmovies{i});
  movieidentifier = n(1:end-9);
  j = find(strcmp(movieidentifiers,movieidentifier));
  assert(numel(j)==1);
  istrkedmovie(j) = true;
end

movieidxtrack = find(istrkedmovie & allpredsfound==1);
order = randperm(numel(movieidxtrack));

for ii = 1:numel(movieidxtrack),
  for p = 1:numel(moviepredfns),
    predfn = moviepredfns{p};
    
    i = movieidxtrack(order(ii));
    switch predfn,
      case 'cpr_2d_locs',
        trxfile = fullfile(moviesavedir,[VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),'_','.mat']);
        [pathcurr,n,~] = fileparts(trxfile);
        resvideo = fullfile(pathcurr,[n,'_TrkVideo.avi']);
      otherwise,
        if isfield(lbld,'finematfiles') && ~isempty(lbld.finematfiles{i}),
          trxfile = lbld.finematfiles{i};
        else
          trxfile = lbld.detectmatfiles{i};
        end
        [~,n,~] = fileparts(trxfile);
        resvideo = fullfile(moviesavedir,[n,'_',predfn,'_TrkVideo.avi']);
    end
    assert(exist(trxfile,'file')>0);
    if exist(resvideo,'file'),
      continue;
    end
    fprintf('ii = %d, i = %d, trxfile = %s, %s\n',ii,i,trxfile,predfn);
    
    %tmp = load(trxfile);
    %nlandmarks = size(tmp.R{1}.cpr_2d_locs,2);
    lkcolors = lines(nlandmarks);
    
    info = mmfileinfo(lbld.movieFilesAll{i,1});
    %figpos = [10,10,nviews*info.Video.Width*2,info.Video.Height*2];
    
    dd = load(trxfile);
    nframes = size(dd.R{1}.(predfn),1);
    if nframes <= nframesplot,
      firstframe = 1;
      endframe = nframes;
    else
      if strcmp(datatype,'jay'),
        firstframe = 251;
        endframe = min(nframes,nframesplot+firstframe-1);
      else
        fmid = round((nframes+1)/2);
        firstframe = fmid-ceil(nframesplot/2);
        endframe = firstframe+nframesplot-1;
      end
    end
    if strcmp(datatype,'jay'),
      info = mmfileinfo(lbld.movieFilesAll{i,1});
      w = info.Video.Width;
      h = info.Video.Height;
      cropframelims = [1,w/2,1,h;w/2+1,w,1,h];
    else
      cropframelims = zeros(0,4);
    end
    
    MakeTrackingResultsHistogramVideo({},trxfile,'moviefilestr','','lkcolors',lkcolors,'PlotTrxLen',0,...
      'TextColor',[.99,.99,.99],'TrxColor','k','figpos',figpos,...
      'resvideo',resvideo,'plotdensity',false,'firstframe',firstframe,'endframe',endframe,'cropframelims',cropframelims,...
      'predfn',predfn);
  end
end
