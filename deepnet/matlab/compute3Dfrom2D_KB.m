function compute3Dfrom2D_KB(savefile,frontviewmatfile,sideviewmatfile,kinematfile,...
  frontviewtrkfile,sideviewtrkfile)

fprintf('Running compute3Dfrom2D_KB with inputs:\n');
fprintf('Front tracked points: %s\n',frontviewmatfile);
fprintf('Side tracked points: %s\n',sideviewmatfile);
fprintf('Calibration info: %s\n',kinematfile);
fprintf('Storing output 3d trajectories to file: %s\n',savefile);
fprintf('Storing front reprojection trajectories to file: %s\n',frontviewtrkfile);
fprintf('Storing side reprojection trajectories to file: %s\n',sideviewtrkfile);


scale_front = 2;
scale_side = 2;
thresh_perc = 99.95;
r_nonmax = 4;

% scale_front = 4;
% scale_side = 4;
% thresh_perc = 99.5;
% r_nonmax = 2;

offx_front = -1;
offy_front = -1;
offx_side = -1;
offy_side = -1;


%% load in tracking results

fprintf('Loading in tracking results...\n');

rdf = load(frontviewmatfile);
rds = load(sideviewmatfile);
FRONTOUTPUTTYPE = 1; % output of MRF
SIDEOUTPUTTYPE = 1; % output of raw detector
% rdf.scores = rdf.scores(:,:,:,:,FRONTOUTPUTTYPE);
% rds.scores = rds.scores(:,:,:,:,SIDEOUTPUTTYPE);
% rdf.locs = permute(rdf.locs(:,:,FRONTOUTPUTTYPE,:),[1,2,4,3]);
% rds.locs = permute(rds.locs(:,:,SIDEOUTPUTTYPE,:),[1,2,4,3]);

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
minscore = -1; %0;
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
if isfield(kd,'DLT_1')
  dlt_side = kd.DLT_1;
  dlt_front = kd.DLT_2;
  useDLT = true;
elseif isfield(kd,'data')
  dlt_side = kd.data.cal.coeff.DLT_1;
  dlt_front = kd.data.cal.coeff.DLT_2;  
  useDLT = true;
else
  calobj = kd.calObj(1);
  useDLT = false;
end



%% 3d reconstruction
% KB: this doesn't seem to be used
% 
% P = nan(3,nlandmarks,Tfront);
% err_reproj = nan(nlandmarks,Tfront);
% predptsfront_re = nan([2,nlandmarks,Tfront]);
% predptsside_re = nan([2,nlandmarks,Tfront]);
% for t = 1:Tfront,
%   for i = 1:nlandmarks,
%     [P(:,i,t),err_reproj(i,t),~,~,predptsfront_re(:,i,t),predptsside_re(:,i,t)] = dlt_2D_to_3D_point(dlt_front,dlt_side,predptsfront(:,i,t),predptsside(:,i,t));
%   end
% end


%% choose a discrete set of samples for each frame

SHOWGMM = false;

thresh_nonmax_front = 0;
thresh_nonmax_side = 0;

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



fprintf('Performing GMM fitting to detection scores...\n');
tic;
for t = 1:Tfront,
  
  if toc > 10,
    fprintf('t = %d\n',t);
    tic;
  end
  
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
    
    end
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

fprintf('Sampling candidate points from GMM fit...\n');

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
    if useDLT
      [Psample(:,1:Kcurr,i,t),w(1:Kcurr,i,t),idxfront,idxside] = Sample3DPoints([xfront{t,i}';yfront{t,i}'],...
        [xside{t,i}';yside{t,i}'],...
        Sfront{t,i},Sside{t,i},dlt_front,dlt_side,wfront{t,i},wside{t,i},Kcurr);
    else
      [Psample(:,1:Kcurr,i,t),w(1:Kcurr,i,t),idxfront,idxside] = Sample3DPoints_orthocam([xfront{t,i}';yfront{t,i}'],...
        [xside{t,i}';yside{t,i}'],...
        Sfront{t,i},Sside{t,i},calobj,wfront{t,i},wside{t,i},Kcurr);
    end
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

fprintf('Choosing the best trajectory through the candidates...\n');

dampen = 0.9;%.5;
poslambdafixed = 1000;
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
if useDLT
  for i = 1:nlandmarks,
    [pfrontbest_re(1,i,:),pfrontbest_re(2,i,:)] = dlt_3D_to_2D(dlt_front,Pbest(1,i,:),Pbest(2,i,:),Pbest(3,i,:));
    [psidebest_re(1,i,:),psidebest_re(2,i,:)] = dlt_3D_to_2D(dlt_side,Pbest(1,i,:),Pbest(2,i,:),Pbest(3,i,:));
  end
else
  for i = 1:nlandmarks
    pfrontbest_re(:,i,:) = calobj.project(squeeze(Pbest(:,i,:)),2);
    psidebest_re(:,i,:) = calobj.project(squeeze(Pbest(:,i,:)),1);
  end    
end

fprintf('Saving results to file %s...\n',savefile);

save(savefile,'pfrontbest','psidebest','Pbest','pfrontbest_re','psidebest_re',...
     'Psample','psample_front','psample_side','w','z','-v7.3');
   
%% convert

fprintf('Saving 2d trx to %s and %s...\n',frontviewtrkfile,sideviewtrkfile);

convertResultsToTrx(savefile,frontviewtrkfile,sideviewtrkfile);


