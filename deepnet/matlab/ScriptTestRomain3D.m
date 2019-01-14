matfiles= {'/home/mayank/temp/romainOut/side1/bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi.mat',...
  '/home/mayank/temp/romainOut/side2/bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi.mat',...
  '/home/mayank/temp/romainOut/bottom/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi.mat'};
calibfiles = '/home/mayank/Dropbox/MultiViewFlyLegTracking/multiview labeling/crig2Optimized_calibjun2916_roiTrackingJun22_20160810_AllExtAllInt.mat';
compute3Dfrom2DRomain('~/temp/aa1.mat',{},...
  matfiles,calibfiles,false);

%%

f = figure;
imagesc(imresize(scorescurr,scale));
hold on;
scatter(mu(:,1),mu(:,2),'+');
for ddd = 1:size(mu,1);
  drawcov(mu(ddd,:),S(:,:,ddd))
  
end

hold off;

%% read the movies

movies = {'/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi'};

readfcn = {};
for mndx = 1:numel(movies);
  readfcn{mndx} = get_readframe_fcn(movies{mndx});
end

%%

ii = {};
for mndx = 1:3
  ii{mndx} = readfcn{mndx}(t);
end


%%

pt = 6;
f = figure;
ccur = xfront(:,t,pt);
rcur = yfront(:,t,pt);

for mndx = 1:3
  subplot(1,3,mndx); 
  imshow(ii{mndx});
  hold on;
  scatter(ccur{mndx},rcur{mndx},'+');
  for ngm = 1:numel(ccur{mndx});
    drawcov( [ccur{mndx}(ngm),rcur{mndx}(ngm)],Sfront{mndx,t,pt}(:,:,ngm));
  end
  hold off;
  
end

%%

pts = 1:6;
cc = hsv(numel(pts));
f = figure;

ii = {};
for t = 1:30
  
  for mndx = 1:3
    ii{mndx} = readfcn{mndx}(t);
    subplot(1,3,mndx); 
    imshow(ii{mndx});
    hold on;
    opts = predpts{mndx}(:,pts,t);
    scatter(opts(1,:),opts(2,:),50,cc,'+');
    npts = pbest_re{mndx}(:,pts,t);
    scatter(npts(2,:),npts(1,:),50,cc,'*');
    hold off;
    title(sprintf('%d',t));
  end
  pause(1);
  
  
end


%% 

movies = {'/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi'};

readfcn = {};
for mndx = 1:numel(movies);
  readfcn{mndx} = get_readframe_fcn(movies{mndx});
end

%%

J = {};
for ndx = 1:numel(matfiles)
  J{ndx} = load(matfiles{ndx});
end

Q = load('~/temp/aa1.mat');
%%

f = figure(320);
pts = 1:6;
cc = jet(numel(pts));
% hax = createsubplots(1,5,.025);

hax = [];
hh = {};
ss = cell(2,3);
t = 1;
for mndx = 1:3
  if mndx<3,
    hax(mndx) = subplot(1,5,mndx);
  else
    hax(mndx) = subplot(1,5,[3,4,5]);
  end
  hold(hax(mndx),'on');
  ii =readfcn{mndx}(t);
  hh{mndx} = imshow(ii,'Parent',hax(mndx));
  ss{1,mndx} = scatter(Q.pbest_re{mndx}(1,pts,t),Q.pbest_re{mndx}(2,pts,t),50,cc,'+','Parent',hax(mndx));
  ss{2,mndx} = scatter(J{mndx}.locs(t,pts,1,1),J{mndx}.locs(t,pts,1,2),50,cc,'*','Parent',hax(mndx));
  hold(hax(mndx),'off');
end


%%
vidobj = VideoWriter('~/temp/aa1.avi');
open(vidobj);
for t = 1:1000
  ii = {};
  for mndx = 1:3
    ii = readfcn{mndx}(t);
    if mndx == 3;
      jj = (log(double(ii(:,:,1))+10)-log(10))/(log(255)-log(10));
      ii = repmat(uint8(jj*255),[1 1 3]);
    end
    set(hh{mndx},'CData',ii);
    set(ss{1,mndx},'XData',Q.pbest_re{mndx}(1,pts,t),'YData',Q.pbest_re{mndx}(2,pts,t));
    set(ss{2,mndx},'XData',J{mndx}.locs(t,pts,1,1),'YData',J{mndx}.locs(t,pts,1,2));
  end
  title(sprintf('%d',t));
  currframe = getframe(f);
  writeVideo(vidobj,currframe);
end

close(vidobj);


%%
movies = {'/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_0_date_2016_06_22_time_11_02_02_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi',...
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi'};
calibfiles = '/home/mayank/Dropbox/MultiViewFlyLegTracking/multiview labeling/crig2Optimized_calibjun2916_roiTrackingJun22_20160810_AllExtAllInt.mat';

readfcn = {};
for mndx = 1:numel(movies);
  readfcn{mndx} = get_readframe_fcn(movies{mndx});
end

%%
t = 143;
pLeft = [122 126];
pBottom = [337 599];
cl = crig.y2x(pLeft,'L');
cb = crig.y2x(pBottom,'B');
[p3l,p3b] = crig.stereoTriangulate(cl,cb,'L','B');
[rleft,cleft] = crig.projectCPR(crig.camxform(p3l,'LL'),1);
[rright,cright] = crig.projectCPR(crig.camxform(p3l,'LR'),2);
[rbottom,cbottom] = crig.projectCPR(crig.camxform(p3l,'LB'),3);

f = figure;
pts = {[rleft,cleft],[rright,cright],[rbottom,cbottom]}; 

ii = {};
for mndx = 1:3
  ii{ndx} = readfcn{mndx}(t);
  subplot(1,3,mndx);
  hold on;
  imshow(ii{ndx});
  scatter(pts{mndx}(2),pts{mndx}(1),50,'+');
%   if mndx ==1,
%     scatter(pLeft(2),pLeft(1),50,'*');
%   elseif mndx ==3
%     scatter(pBottom(2),pBottom(1),50,'*');
%   end  
  hold off;
end
