function tracks = compute3Dfrom2DRomain(vscores,cRig)

% vscores should be a cell with dims TxYxXxPts

% Order of views is Left,Right and then Bottom for Romain

offx_front = 0;
offy_front = 0;
offx_side = 0;
offy_side = 0;
scale= 4;

nviews = numel(vscores);

thresh_nonmax_front = 0;
thresh_nonmax_side = 0;
thresh_perc = 99.7;
r_nonmax = 2;

[T,~,~,nlandmarks] = size(vscores{1});
Tfront = T;

%% 3D reconstruction

xfront = cell(nviews,Tfront,nlandmarks);
yfront = cell(nviews,Tfront,nlandmarks);
wfront = cell(nviews,Tfront,nlandmarks);
Sfront = cell(nviews,Tfront,nlandmarks);

for t = 1:Tfront,
  
  for i = 1:nlandmarks,
  
    for view = 1:nviews,
      
      [xgrid_front,ygrid_front] = meshgrid(((1:size(vscores{view},3))+offx_front)*scale,...
        ((1:size(vscores{view},2))+offy_front)*scale);

      scorescurr = permute(vscores{view}(t,:,:,i),[2,3,1]);
      tscores = scorescurr(r_nonmax+1:end-r_nonmax,r_nonmax+1:end-r_nonmax); % safeguard against boundaries
      threshcurr = prctile(tscores(:),thresh_perc);
      offx = offx_front;
      offy = offy_front;
      xgrid = xgrid_front;
      ygrid = ygrid_front;

      tscorescurr = scorescurr;
      minscores = min(scorescurr(:));
      tscorescurr(1:r_nonmax,:) = minscores;
      tscorescurr(end-r_nonmax+1:end,:) = minscores;
      tscorescurr(:,1:r_nonmax) = minscores;
      tscorescurr(:,end-r_nonmax+1:end) = minscores;
      
      [r,c] = nonmaxsuppts(tscorescurr,r_nonmax,threshcurr);
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
      
      xfront{view,t,i} = mu(:,1);
      yfront{view,t,i} = mu(:,2);
      Sfront{view,t,i} = S;
      wfront{view,t,i} = w;
    
    end
  end
  if mod(t,50)==0, fprintf('.'); end
  if mod(t,1000)==0, fprintf('\n'); end
end
fprintf('\n');
%%

K = 50;
Kneighbor = 10;
discountneighbor = .05;

% J = load(calibfile);
% cRig = J.crig2AllExtAllInt;
% cRig = CalibratedRig2(calibfiles{1},calibfiles{2});

Psample = zeros([3,K+2*Kneighbor,nlandmarks,T]);
w = zeros([K+2*Kneighbor,nlandmarks,T]);
nsamples = zeros(T,nlandmarks);
fprintf('Generating candidates\n');
for t = 1:T,
  for i = 1:nlandmarks,

    if t == 1 || t == T,
      Kcurr = K + Kneighbor;
    else
      Kcurr = K;
    end
    
    s1 = size(xfront{1,t,i},1);
    s2 = size(xfront{2,t,i},1);
    s3 = size(xfront{3,t,i},1);
    Kcurr = min(Kcurr,s1*s2+s2*s3+s3*s1);
    ccur = xfront(:,t,i);
    rcur = yfront(:,t,i);
    [Psample(:,1:Kcurr,i,t),w(1:Kcurr,i,t)] = ...
      Sample3DPointsRomain(rcur,ccur,Sfront(:,t,i),wfront(:,t,i),cRig,Kcurr);
    nsamples(t,i) = Kcurr;
  end
  if mod(t,50)==0, fprintf('.'); end
  if mod(t,1000)==0, fprintf('\n'); end
end
fprintf('\n');

% add the top Kneighbor points from adjacent frames, which corresponds to
% just interpolating through the current frame
for t = 1:T,
  for i = 1:nlandmarks,
    if t < T,
      Kcurr = min(Kneighbor,nsamples(t+1,i));
      Psample(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = Psample(:,1:Kcurr,i,t+1);
      w(nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = w(1:Kcurr,i,t+1)*discountneighbor;
      nsamples(t,i) = nsamples(t,i)+Kcurr;
    end
    if t > 1,
      Kcurr = min(Kneighbor,nsamples(t-1,i));
      Psample(:,nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = Psample(:,1:Kcurr,i,t-1);
      w(nsamples(t,i)+1:nsamples(t,i)+Kcurr,i,t) = w(1:Kcurr,i,t-1)*discountneighbor;
      nsamples(t,i) = nsamples(t,i)+Kcurr;
    end
  end
end

maxnsamples = find(any(any(w>0,2),3),1,'last');
Psample = Psample(:,1:maxnsamples,:,:);
w = w(1:maxnsamples,:,:);

z = sum(w,1);
w = bsxfun(@rdivide,w,z);

%% choose trajectory

dampen = .9;
poslambdafixed = 1000;
Pbest = nan(3,nlandmarks,T);
cost = nan(1,nlandmarks);
poslambda = nan(1,nlandmarks);
idx = nan(nlandmarks,T);
for i = 1:nlandmarks
  fprintf('Landmark %d / %d\n',i,nlandmarks);
  X = permute(Psample(:,:,i,:),[1,4,2,3]);
  appearancecost = permute(w(:,i,:),[3,1,2]);
  [Xbest,idxcurr,totalcost,poslambdacurr] = ChooseBestTrajectory(X,-log(appearancecost),'dampen',dampen,'poslambda',poslambdafixed);
  % Xbest is D x T
  Pbest(:,i,:) = Xbest;
  cost(i) = totalcost;
  poslambda(i) = poslambdacurr;
  idx(i,:) = idxcurr;
end

cNames = {'L','R','B'};
pbest_re = cell(1,3);
for view = 1:nviews
  pbest_re{view} = nan([2,nlandmarks,T]);
  for i = 1:nlandmarks
    pview = cRig.camxform(squeeze(Pbest(:,i,:)),[cNames{1} cNames{view}]);
    [pbest_re{view}(2,i,:),pbest_re{view}(1,i,:)] = cRig.projectCPR(pview,view);
  end
end

tracks = struct;
tracks.Pbest = Pbest;
tracks.pbest_re = pbest_re;
tracks.Psample = Psample;
tracks.w = w;
tracks.z = z;
% save(savefile,'Pbest','pbest_re', 'Psample','w','z','-v7.3');

%% plot results with temporal smoothing

dosave = false;
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

function [P_all,w_all] = Sample3DPointsRomain(r,c,S,w_in,cRig,K)
                        
  view2skip = 6;
  nviews = numel(r);
  cNames = {'L','R','B'}; % for now.
  P_all = nan(3,numel(r{1})+1,numel(r{2})+1,numel(r{3})+1,3);
  w_all = zeros(numel(r{1})+1,numel(r{2})+1,numel(r{3})+1,3);
  for view = 1:numel(r),
    if view == view2skip, continue; end
    fview = view;
    sview = mod(view,nviews)+1;
    oview = mod(view+1,nviews)+1;
    rfront = r{fview};
    rside = r{sview};
    rother = r{oview};
    cfront = c{fview};
    cside = c{sview};
    cother = c{oview};
    
    nfront = numel(cfront);
    nside = numel(cside);
    nother = numel(cother);
    w = zeros(nfront+1,nside+1,nother+1);
    pfront = nan(nfront,nside);
    pside = nan(nfront,nside);
    pother = nan(nfront,nside);
    P = nan(3,nfront+1,nside+1,nother+1);
    Sfront = S{fview};
    Sside  = S{sview};
    Sother  = S{oview};
    wfront = w_in{fview};
    wside  = w_in{sview};
    wother = w_in{oview};
    for ifront = 1:nfront,
      for iside = 1:nside,
%         [P(:,ifront,iside),~,~,~,xfront_re,xside_re] = dlt_2D_to_3D_point(dlt_front,dlt_side,xfront(:,ifront),xside(:,iside),...
%           'Sfront',Sfront(:,:,ifront),'Sside',Sside(:,:,iside));
%         pfront(ifront,iside) = mvnpdf(xfront_re,xfront(:,ifront)',Sfront(:,:,ifront));
%         pside(ifront,iside) = mvnpdf(xside_re,xside(:,iside)',Sside(:,:,iside));
%         w(ifront,iside) = pfront(ifront,iside)*pside(ifront,iside)*wfront(ifront)*wside(iside);

        cropfront = cRig.y2x([rfront(ifront) cfront(ifront)],cNames{fview});
        cropside = cRig.y2x([rside(iside) cside(iside)],cNames{sview});
        [P3d_front,P3d_side] = cRig.stereoTriangulate(cropfront,cropside,cNames{fview},cNames{sview});
        [rfront_re,cfront_re] = cRig.projectCPR(P3d_front,fview);
        [rside_re,cside_re] = cRig.projectCPR(P3d_side,sview);
        [rother_re,cother_re] = cRig.projectCPR(cRig.camxform(P3d_front,[cNames{fview} cNames{oview}]),oview);
        
        pfront(ifront,iside) = mvnpdf([cfront_re rfront_re],[cfront(ifront) rfront(ifront)],Sfront(:,:,ifront));
        pside(ifront,iside) = mvnpdf([cside_re rside_re],[cside(iside) rside(iside)],Sside(:,:,iside));
        
        oprobs = zeros(1,numel(cother)+1);
        for iother = 1:numel(cother)
          oprobs(iother) = mvnpdf([cother_re rother_re],[rother(iother) cother(iother)],Sother(:,:,iother));
        end
        oprobs(end) = 0.000001;  % in case the point isn't visible in the "other" view.
        [pother(ifront,iside),idxother] = max(oprobs);
        w(ifront,iside,idxother) = pfront(ifront,iside)*pside(ifront,iside)*wfront(ifront)*wside(iside)*pother(ifront,iside);
        curP = cRig.camxform(P3d_front,[cNames{fview} cNames{1}]);
        P(:,ifront,iside,idxother) = curP;
      end
    end
    for ixx = 1:(view-1)
      P = permute(P,[1 4 2 3]);
      w = permute(w,[3 1 2]);
    end
    P_all(:,:,:,:,view) = P;
    w_all(:,:,:,view) = w;
  end
  
  w_all = w_all / nansum(w_all(:));
  [~,order] = sort(w_all(:),1,'descend');

  P_all = P_all(:,order(1:K));
  w_all = w_all(order(1:K));
