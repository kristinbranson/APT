function compute3Dfrom2D(savefile,frontviewvideofile,sideviewvideofile,...
  frontviewmatfile,sideviewmatfile,kinematfile,dosave,experiment_name)
% outdated. moving to maintaining only _KB.

compute3Dfrom2D_KB(savefile,frontviewmatfile,sideviewmatfile,kinematfile,...
    '/tmp/front.trk','/tmp/side.trk');

return;

%%

offx_front = -1;
offy_front = -1;
offx_side = -1;
offy_side = -1;
scale_front = 4;
scale_side = 4;
front_thresh = 0.30;
side_thresh = 0.30;

%% load in tracking results

rdf = load(frontviewmatfile);
rds = load(sideviewmatfile);
FRONTOUTPUTTYPE = 1; % output of MRF
SIDEOUTPUTTYPE = 1; % output of raw detector
rdf.scores = rdf.scores(:,:,:,:,FRONTOUTPUTTYPE);
rds.scores = rds.scores(:,:,:,:,SIDEOUTPUTTYPE);
rdf.locs = permute(rdf.locs(:,:,FRONTOUTPUTTYPE,:),[1,2,4,3]);
rds.locs = permute(rds.locs(:,:,SIDEOUTPUTTYPE,:),[1,2,4,3]);

v0 = rdf.scores(1);

isdata = squeeze(any(any(rdf.scores~=v0,1),4));
tmp = any(isdata,2);
miny_front = find(tmp,1);
maxy_front = find(tmp,1,'last');
tmp = any(isdata,1);
minx_front = find(tmp,1);
maxx_front = find(tmp,1,'last');

% alternatively by Mayank
% Front is MRF so the scores go from 0 to 1
miny_front = 2;
maxy_front = size(rdf.scores,2);
minx_front = 2;
maxx_front = size(rdf.scores,3);
minscore = 0;
maxscore = 1;

rdf.scores(:,miny_front:maxy_front,minx_front:maxx_front,:) = ...
  (rdf.scores(:,miny_front:maxy_front,minx_front:maxx_front,:) - minscore)/(maxscore-minscore);


v0 = rds.scores(1);
isdata = squeeze(any(any(rds.scores~=v0,1),4));
tmp = any(isdata,2);
miny_side = find(tmp,1);
maxy_side = find(tmp,1,'last');
tmp = any(isdata,1);
minx_side = find(tmp,1);
maxx_side = find(tmp,1,'last');

% alternatively by Mayank
% Side is raw so the scores go from -1 to 1
miny_side = 2;
maxy_side = size(rds.scores,2);
minx_side = 2;
maxx_side = size(rds.scores,3);
minscore = -1;
maxscore = 1;
rds.scores(:,miny_side:maxy_side,minx_side:maxx_side,:) = ...
  (rds.scores(:,miny_side:maxy_side,minx_side:maxx_side,:) - minscore)/(maxscore-minscore);

% locs is nframes x nlandmarks x noutputtypes x ndims
% scores is nframes x 256 x 256 x nlandmarks x noutputtypes

predptsfront = rdf.locs;
predptsfront = permute(predptsfront,[3,2,1]);
predptsside = rds.locs;
predptsside = permute(predptsside,[3,2,1]);


[~,nlandmarks,Tfront] = size(predptsfront);
[~,~,Tside] = size(predptsside);
assert(Tfront==Tside);
T = Tfront;

kd = load(kinematfile);
if isfield(kd,'data'),
  dlt_side = kd.data.cal.coeff.DLT_1;
  dlt_front = kd.data.cal.coeff.DLT_2;
else
  dlt_side = kd.DLT_1;
  dlt_front = kd.DLT_2;
end



%% 3d reconstruction

P = nan(3,nlandmarks,Tfront);
err_reproj = nan(nlandmarks,Tfront);
predptsfront_re = nan([2,nlandmarks,Tfront]);
predptsside_re = nan([2,nlandmarks,Tfront]);
for t = 1:Tfront,
  for i = 1:nlandmarks,
    [P(:,i,t),err_reproj(i,t),~,~,predptsfront_re(:,i,t),predptsside_re(:,i,t)] = dlt_2D_to_3D_point(dlt_front,dlt_side,predptsfront(:,i,t),predptsside(:,i,t));
  end
end


%% choose a discrete set of samples for each frame

SHOWGMM = false;

thresh_nonmax_front = 0;
thresh_nonmax_side = 0;
thresh_perc = 99.5;
r_nonmax = 2;

[dx,dy] = meshgrid(-r_nonmax:r_nonmax,-r_nonmax:r_nonmax);
fil = double(dx.^2 + dy.^2 <= r_nonmax^2);
 
% [~,nr_front,nc_front,~] = size(rdf.scores);
% [~,nr_side,nc_side,~] = size(rds.scores);

[xgrid_front,ygrid_front] = meshgrid(((minx_front:maxx_front)+offx_front)*scale_front,...
  ((miny_front:maxy_front)+offy_front)*scale_front);
[xgrid_side,ygrid_side] = meshgrid(((minx_side:maxx_side)+offx_side)*scale_side,...
  ((miny_side:maxy_side)+offy_side)*scale_side);

xfront = cell(Tfront,nlandmarks);
yfront = cell(Tfront,nlandmarks);
wfront = cell(Tfront,nlandmarks);
Sfront = cell(Tfront,nlandmarks);
xside = cell(Tfront,nlandmarks);
yside = cell(Tfront,nlandmarks);
wside = cell(Tfront,nlandmarks);
Sside = cell(Tfront,nlandmarks);

hfig = 1;
figure(hfig);
clf;
colors = jet(nlandmarks);
if SHOWGMM,
  [readframe_front,~] = get_readframe_fcn(frontviewvideofile);
  [readframe_side,~] = get_readframe_fcn(sideviewvideofile);

  hax = createsubplots(2,1+nlandmarks,.05);
  hax = reshape(hax,[2,1+nlandmarks]);
  him = nan(2,1+nlandmarks);
  hx1 = nan(2,1+nlandmarks);
  hx2 = cell(2,1+nlandmarks);
  hti = nan;

  scatterscalefactor = 5;
end

for t = 1:Tfront,
  
  for i = 1:nlandmarks,
  
    for view = 1:2,
      
      if view == 1,    
        scorescurr = permute(rdf.scores(t,miny_front:maxy_front,minx_front:maxx_front,i),[2,3,1]);
%         threshcurr = thresh_nonmax_front;
        tscores = scorescurr(r_nonmax+1:end-r_nonmax,r_nonmax+1:end-r_nonmax); % safeguard against boundaries
        threshcurr = prctile(tscores(:),thresh_perc);
        miny = miny_front;
        maxy = maxy_front;
        minx = minx_front;
        maxx = maxx_front;
        scale = scale_front;
        offx = offx_front;
        offy = offy_front;
        xgrid = xgrid_front;
        ygrid = ygrid_front;
      else
        scorescurr = permute(rds.scores(t,miny_side:maxy_side,minx_side:maxx_side,i),[2,3,1]);
%         threshcurr = thresh_nonmax_side;
        tscores = scorescurr(r_nonmax+1:end-r_nonmax,r_nonmax+1:end-r_nonmax); % safeguard against boundaries
        threshcurr = prctile(tscores(:),thresh_perc);
        miny = miny_side;
        maxy = maxy_side;
        minx = minx_side;
        maxx = maxx_side;
        scale = scale_side;
        offx = offx_side;
        offy = offy_side;
        xgrid = xgrid_side;
        ygrid = ygrid_side;
      end

      tscorescurr = scorescurr;
      minscores = min(scorescurr(:));
      tscorescurr(1:r_nonmax,:) = minscores;
      tscorescurr(end-r_nonmax+1:end,:) = minscores;
      tscorescurr(:,1:r_nonmax) = minscores;
      tscorescurr(:,end-r_nonmax+1:end) = minscores;
      
      [r,c] = nonmaxsuppts(tscorescurr,r_nonmax,threshcurr);
      r = r + miny - 1;
      c = c + minx - 1;
      r = (r+offy)*scale;
      c = (c+offx)*scale;
      idxcurr = scorescurr >= threshcurr;
    
      k0 = numel(r);
      k = k0+min(numel(r),floor(nnz(idxcurr)/4));
      %k = 2*k0;
      start = nan(k,2);
      start(1:k0,:) = [c(:),r(:)];
      X = [xgrid(idxcurr),ygrid(idxcurr)];
      d = min(dist2(X,start(1:k0,:)),[],2);
      for j = k0+1:k,
        [~,maxj] = max(d);
        start(j,:) = X(maxj,:);
        d = min(d,dist2(X,start(j,:)));
      end
      [mu,S,~,post] = mygmm(X,k,...
        'Start',start,...
        'weights',scorescurr(idxcurr)-threshcurr);
      w = sum(bsxfun(@times,scorescurr(idxcurr),post),1)';
      w(w<0.1) = 0.1;
      
      nanmu = any(isnan(mu),2);
      mu = mu(~nanmu,:);
      w = w(~nanmu);
      S = S(:,:,~nanmu);
      
      if view == 1,
        xfront{t,i} = mu(:,1);
        yfront{t,i} = mu(:,2);
        Sfront{t,i} = S;
        wfront{t,i} = w;
      else
        xside{t,i} = mu(:,1);
        yside{t,i} = mu(:,2);
        Sside{t,i} = S;
        wside{t,i} = w;
      end
      if SHOWGMM,
        if ishandle(him(view,1+i)),
          set(him(view,1+i),'CData',scorescurr);
        else
          hold(hax(view,1+i),'off');
          him(view,i+1) = imagesc([minx+offx,maxx+offx]*scale,...
            [miny+offy,maxy+offy]*scale,...
            scorescurr,'Parent',hax(view,1+i));
          axis(hax(view,1+i),'image','off');
          hold(hax(view,1+i),'on');
        end
      end
    
    end
  end

  if SHOWGMM,
    if ishandle(him(2,1)),
      set(him(2,1),'CData',readframe_side(t));
    else
      hold(hax(2,1),'off');
      image(readframe_side(t),'Parent',hax(2,1));
      axis(hax(2,1),'image','off');
      hold(hax(2,1),'on');
      set(hax(2,1),'XLim',...
        ([minx_side,maxx_side]+offx_side)*scale_side,...
        'YLim',([miny_side,maxy_side]+offy_side)*scale_side);
    end

    if ishandle(him(1,1)),
      set(him(1,1),'CData',readframe_front(t));
    else
      hold(hax(1,1),'off');
      image(readframe_front(t),'Parent',hax(1,1));
      axis(hax(1,1),'image','off');
      hold(hax(1,1),'on');
      set(hax(1,1),'XLim',([minx_front,maxx_front]+offx_front)*scale_front,...
        'YLim',([miny_front,maxy_front]+offy_front)*scale_front);
    end
    if ishandle(hti),
      set(hti,'String',sprintf('%d / %d',t,Tfront));
    else
      hti = title(hax(1,1),sprintf('%d / %d',t,Tfront));
    end


    for i = 1:nlandmarks,
      if ishandle(hx1(1,1+i)),
        delete(hx1(1,1+i));
      end
      hx1(1,1+i) = scatter(xfront{t,i},yfront{t,i},wfront{t,i}*scatterscalefactor,colors(i,:),'+','Parent',hax(1,1));
      if ishandle(hx1(2,1+i)),
        delete(hx1(2,1+i));
      end
      hx1(2,1+i) = scatter(xside{t,i},yside{t,i},wside{t,i}*scatterscalefactor,colors(i,:),'+','Parent',hax(2,1));
    end

    for i = 1:nlandmarks,

      delete(hx2{1,1+i}(ishandle(hx2{1,1+i})));
      hx2{1,1+i} = nan(1,numel(xfront{t,i}));
      for j = 1:numel(xfront{t,i}),
        hx2{1,1+i}(j) = drawcov([xfront{t,i}(j),yfront{t,i}(j)],Sfront{t,i}(:,:,j),'Parent',hax(1,1+i),'Color','k','LineWidth',2);
      end

      delete(hx2{2,1+i}(ishandle(hx2{2,1+i})));
      hx2{2,1+i} = nan(1,numel(xside{t,i}));
      for j = 1:numel(xside{t,i}),
        hx2{2,1+i}(j) = drawcov([xside{t,i}(j),yside{t,i}(j)],Sside{t,i}(:,:,j),'Parent',hax(2,1+i),'Color','k','LineWidth',2);
      end

    end

    drawnow;
  end
end

%%

% % test with just one point for now -- get mixture of Gaussians from Mayank!
% xfront = permute(predptsfront,[1,4,2,3]);
% xside = permute(predptsside,[1,4,2,3]);
% wfront = ones([1,nlandmarks,Tfront])/nlandmarks;
% wside = ones([1,nlandmarks,Tfront])/nlandmarks;
% Sfront = repmat(eye(2)*4^2,[1,1,1,nlandmarks,T]);
% Sside = repmat(eye(2)*4^2,[1,1,1,nlandmarks,T]);

K = 50;
Kneighbor = 10;
% K = 1;
% Kneighbor = 1;
discountneighbor = .05;

Psample = nan([3,K+2*Kneighbor,nlandmarks,T]);
psample_front = nan([2,K+2*Kneighbor,nlandmarks,T]);
psample_side = nan([2,K+2*Kneighbor,nlandmarks,T]);
w = zeros([K+2*Kneighbor,nlandmarks,T]);
nsamples = zeros(T,nlandmarks);
for t = 1:T,
  for i = 1:nlandmarks,

    if t == 1 || t == T,
      Kcurr = K + Kneighbor;
    else
      Kcurr = K;
    end
    Kcurr = min(Kcurr,size(xfront{t,i},1)*size(xside{t,i},1));
    [Psample(:,1:Kcurr,i,t),w(1:Kcurr,i,t),idxfront,idxside] = Sample3DPoints([xfront{t,i}';yfront{t,i}'],...
      [xside{t,i}';yside{t,i}'],...
      Sfront{t,i},Sside{t,i},dlt_front,dlt_side,wfront{t,i},wside{t,i},Kcurr);
    psample_front(1,1:Kcurr,i,t) = xfront{t,i}(idxfront);
    psample_front(2,1:Kcurr,i,t) = yfront{t,i}(idxfront);
    psample_side(1,1:Kcurr,i,t) = xside{t,i}(idxside);
    psample_side(2,1:Kcurr,i,t) = yside{t,i}(idxside);
    nsamples(t,i) = Kcurr;
    
  end
end

% add the top Kneighbor points from adjacent frames, which corresponds to
% just interpolating through the current frame
for t = 1:T,
  for i = 1:nlandmarks,
    if t < T,
      Kcurr = min(Kneighbor,nsamples(t+1,i));
      Psample(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = Psample(:,1:Kcurr,i,t+1);
      w(nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = w(1:Kcurr,i,t+1)*discountneighbor;
      psample_front(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = psample_front(:,1:Kcurr,i,t+1);
      psample_side(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = psample_side(:,1:Kcurr,i,t+1);
      nsamples(t,i) = nsamples(t,i)+Kcurr;
    end
    if t > 1,
      Kcurr = min(Kneighbor,nsamples(t-1,i));
      Psample(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = Psample(:,1:Kcurr,i,t-1);
      w(nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = w(1:Kcurr,i,t-1)*discountneighbor;
      psample_front(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = psample_front(:,1:Kcurr,i,t-1);
      psample_side(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = psample_side(:,1:Kcurr,i,t-1);
      nsamples(t,i) = nsamples(t,i)+Kcurr;
    end
  end
end

maxnsamples = find(any(any(w>0,2),3),1,'last');
Psample = Psample(:,1:maxnsamples,:,:);
psample_front = psample_front(:,1:maxnsamples,:,:);
psample_side = psample_side(:,1:maxnsamples,:,:);
w = w(1:maxnsamples,:,:);

z = sum(w,1);
w = bsxfun(@rdivide,w,z);

%% choose trajectory

dampen = .9;
poslambdafixed = 1000;%100; before 20170322 -- Mayank
Pbest = nan(3,nlandmarks,T);
cost = nan(1,nlandmarks);
poslambda = nan(1,nlandmarks);
idx = nan(nlandmarks,T);
pfrontbest = nan(2,nlandmarks,T);
psidebest = nan(2,nlandmarks,T);
for i = 1:nlandmarks,
  fprintf('Landmark %d / %d\n',i,nlandmarks);
  X = permute(Psample(:,:,i,:),[1,4,2,3]);
  appearancecost = permute(w(:,i,:),[3,1,2]);
  [Xbest,idxcurr,totalcost,poslambdacurr] = ChooseBestTrajectory(X,-log(appearancecost),'dampen',dampen,'poslambda',poslambdafixed);
  % Xbest is D x T
  Pbest(:,i,:) = Xbest;
  for t = 1:T,
    pfrontbest(:,i,t) = psample_front(:,idxcurr(t),i,t);
    psidebest(:,i,t) = psample_side(:,idxcurr(t),i,t);
  end
  cost(i) = totalcost;
  poslambda(i) = poslambdacurr;
  idx(i,:) = idxcurr;
end

pfrontbest_re = nan([2,nlandmarks,T]);
psidebest_re = nan([2,nlandmarks,T]);
for i = 1:nlandmarks,
  [pfrontbest_re(1,i,:),pfrontbest_re(2,i,:)] = dlt_3D_to_2D(dlt_front,Pbest(1,i,:),Pbest(2,i,:),Pbest(3,i,:));
  [psidebest_re(1,i,:),psidebest_re(2,i,:)] = dlt_3D_to_2D(dlt_side,Pbest(1,i,:),Pbest(2,i,:),Pbest(3,i,:));
end

save(savefile,'pfrontbest','psidebest','Pbest','pfrontbest_re','psidebest_re',...
     'Psample','psample_front','psample_side','w','z','-v7.3');

%% plot results with temporal smoothing

if dosave,
  [readframe_front,~] = get_readframe_fcn(frontviewvideofile);
  [readframe_side,~] = get_readframe_fcn(sideviewvideofile);

  colorscalefactor = 1;
  dosave = true;
  if dosave,
  %   vidobj = VideoWriter(sprintf('../CNNTrackingResults20160425/3DRecon_%s.avi',experiment_name));
    vidobj = VideoWriter(sprintf('%s_video.avi',savefile));
    open(vidobj);
  end

  hfig = 1;
  figure(hfig);
  clf;
  hax = createsubplots(2,3,.025);
  hax = reshape(hax,[2,3]);
  delete(hax(2,3));

  for t = 1:Tfront,

    im_front = readframe_front(t);
    im_side = readframe_side(t);

    nr = maxy_front-miny_front+1;
    nc = maxx_front-minx_front+1;
    if size(im_front,3)==1
      im_front = repmat(im_front,[1 1 3]);
    end
    if size(im_side,3)==1
      im_side = repmat(im_side,[1 1 3]);
    end
    scoreim_front = double(reshape(im_front(((miny_front:maxy_front)+offy_front)*scale_front,...
      ((minx_front:maxx_front)+offx_front)*scale_front,:),[nr*nc,3]))/255/2;
    %scoreim_front = zeros(nr*nc,3);
    for i = 1:nlandmarks,
      scorescurr = permute(rdf.scores(t,miny_front:maxy_front,minx_front:maxx_front,i),[2,3,1]);
      scorescurr = max(0,min(1,colorscalefactor*(scorescurr - thresh_nonmax_front)/(max(scorescurr(:))-thresh_nonmax_front)));
      scoreim_front = scoreim_front + bsxfun(@times,colors(i,:),scorescurr(:));
    end
    scoreim_front = min(scoreim_front,1);
    %scoreim_front = scoreim_front/nlandmarks;
    scoreim_front = reshape(scoreim_front,[nr,nc,3]);

    nr = maxy_side-miny_side+1;
    nc = maxx_side-minx_side+1;
    scoreim_side = double(reshape(im_side(((miny_side:maxy_side)+offy_side)*scale_side,...
      ((minx_side:maxx_side)+offx_side)*scale_side,:),[nr*nc,3]))/255/2;
    %scoreim_side = zeros(nr*nc,3);
    for i = 1:nlandmarks,
      scorescurr = permute(rds.scores(t,miny_side:maxy_side,minx_side:maxx_side,i),[2,3,1]);
      scorescurr = max(0,min(1,colorscalefactor*(scorescurr - thresh_nonmax_side)/(max(scorescurr(:))-thresh_nonmax_side)));
      scoreim_side = scoreim_side + bsxfun(@times,colors(i,:),scorescurr(:));
    end
    scoreim_side = min(scoreim_side,1);
    %scoreim_side = scoreim_side/nlandmarks;
    scoreim_side = reshape(scoreim_side,[nr,nc,3]);

    if t == 1,
      him_front = imshow(im_front,'Parent',hax(1,1));
      axis(hax(1,1),'image','off');
      set(hax(1,1),'XLim',([minx_front,maxx_front]+offx_front)*scale_front,...
        'YLim',([miny_front,maxy_front]+offy_front)*scale_front);
      hold(hax(1,1),'on');

      hscore_front = imshow(scoreim_front,'Parent',hax(2,1));
      hold(hax(2,1),'on');
      axis(hax(2,1),'image','off');

      hfrontx = nan(1,nlandmarks);
      for i = 1:nlandmarks,
        hfrontx(i) = plot(hax(1,1),pfrontbest_re(1,i,t),pfrontbest_re(2,i,t),'x','Color',colors(i,:),'MarkerSize',8);
      end
      hfront = nan(1,nlandmarks);
      for i = 1:nlandmarks,
        hfront(i) = plot(hax(1,1),[pfrontbest_re(1,i,t),pfrontbest(1,i,t)],...
          [pfrontbest_re(2,i,t),pfrontbest(2,i,t)],'-','Color',colors(i,:));
      end

      hfrontx2 = plot(hax(2,1),pfrontbest_re(1,:,t)/scale_front-offx_front-minx_front+1,pfrontbest_re(2,:,t)/scale_front-offy_front-miny_front+1,'wx');
      hfrontx3 = plot(hax(2,1),pfrontbest_re(1,:,t)/scale_front-offx_front-minx_front+1,pfrontbest_re(2,:,t)/scale_front-offy_front-miny_front+1,'k+');

      him_side = imshow(im_side,'Parent',hax(1,2));
      axis(hax(1,2),'image','off');
      set(hax(1,2),'XLim',...
        ([minx_side,maxx_side]+offx_side)*scale_side,...
        'YLim',([miny_side,maxy_side]+offy_side)*scale_side);
      hold(hax(1,2),'on');

      hscore_side = imshow(scoreim_side,'Parent',hax(2,2));
      hold(hax(2,2),'on');
      axis(hax(2,2),'image','off');

      hsidex = nan(1,nlandmarks);
      for i = 1:nlandmarks,
        hsidex(i) = plot(hax(1,2),psidebest_re(1,i,t),psidebest_re(2,i,t),'x','Color',colors(i,:),'MarkerSize',8);
      end
      hside = nan(1,nlandmarks);
      for i = 1:nlandmarks,
        hside(i) = plot(hax(1,2),[psidebest_re(1,i,t),psidebest(1,i,t)],...
          [psidebest_re(2,i,t),psidebest(2,i,t)],'-','Color',colors(i,:));
      end
      hsidex2 = plot(hax(2,2),psidebest_re(1,:,t)/scale_side-offx_side-minx_side+1,psidebest_re(2,:,t)/scale_side-offy_side-miny_side+1,'wx');
      hsidex3 = plot(hax(2,2),psidebest_re(1,:,t)/scale_side-offx_side-minx_side+1,psidebest_re(2,:,t)/scale_side-offy_side-miny_side+1,'k+');

      h3 = nan(1,nlandmarks);
      for i = 1:nlandmarks,
        h3(i) = plot3(hax(1,3),Pbest(1,i,t),Pbest(2,i,t),Pbest(3,i,t),'x','Color',colors(i,:),'LineWidth',2);
        if i == 1,
          hold(hax(1,3),'on');
        end
      end
      minP = min(min(Pbest,[],2),[],3);
      maxP = max(max(Pbest,[],2),[],3);
      dP = maxP-minP;
      axis(hax(1,3),'equal');
      set(hax(1,3),'XLim',[minP(1)-dP(1)/20,maxP(1)+dP(1)/20],...
        'YLim',[minP(2)-dP(2)/20,maxP(2)+dP(2)/20],...
        'ZLim',[minP(3)-dP(3)/20,maxP(3)+dP(3)/20]);
      xlabel(hax(1,3),'X');
      ylabel(hax(1,3),'Y');
      zlabel(hax(1,3),'Z');
      grid(hax(1,3),'on');

    else
      set(him_front,'CData',im_front);
      set(him_side,'CData',im_side);
      set([hfrontx2,hfrontx3],'XData',pfrontbest_re(1,:,t)/scale_front-offx_front-minx_front+1,'YData',pfrontbest_re(2,:,t)/scale_front-offy_front-miny_front+1);
      set([hsidex2,hsidex3],'XData',psidebest_re(1,:,t)/scale_side-offx_side-minx_side+1,'YData',psidebest_re(2,:,t)/scale_side-offy_side-miny_side+1);

      for i = 1:nlandmarks,
        set(hfront(i),'XData',[pfrontbest_re(1,i,t),pfrontbest(1,i,t)],...
          'YData',[pfrontbest_re(2,i,t),pfrontbest(2,i,t)]);
        set(hfrontx(i),'XData',pfrontbest_re(1,i,t),'YData',pfrontbest_re(2,i,t));
        set(hside(i),'XData',[psidebest_re(1,i,t),psidebest(1,i,t)],...
          'YData',[psidebest_re(2,i,t),psidebest(2,i,t)]);
        set(hsidex(i),'XData',psidebest_re(1,i,t),'YData',psidebest_re(2,i,t));
        set(h3(i),'XData',Pbest(1,i,t),'YData',Pbest(2,i,t),'ZData',Pbest(3,i,t));
        plot3(hax(1,3),squeeze(Pbest(1,i,t-1:t)),squeeze(Pbest(2,i,t-1:t)),squeeze(Pbest(3,i,t-1:t)),'-','Color',colors(i,:));
        set(hscore_front,'CData',scoreim_front);
        set(hscore_side,'CData',scoreim_side);
      end

    end

    if dosave,
      drawnow;
      fr = getframe(hfig);
      writeVideo(vidobj,fr);
    else
      pause(.1);
    end

  end

  if dosave,
    close(vidobj);
  end
end
