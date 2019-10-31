sidemoviefile = 'calibration/cali_side_v001.avi';
frontmoviefile = 'calibration/cali_front_v001.avi';
combmoviefile = 'calibration/cali_comb.avi';
nframessample = 5;
outdir = 'calibration/calibims20150212';

% %% concatenate
% 
% catmov(sidemoviefile,frontmoviefile,combmoviefile);

%% set up directories

if ~exist(outdir,'dir'),
  mkdir(outdir);
end
if ~exist(fullfile(outdir,'side'),'dir'),
  mkdir(fullfile(outdir,'side'));
else
  unix(sprintf('rm %s/side*tif',fullfile(outdir,'side')));
end
if ~exist(fullfile(outdir,'front'),'dir'),
  mkdir(fullfile(outdir,'front'));
else
  unix(sprintf('rm %s/front*tif',fullfile(outdir,'front')));
end

%% select frames where grid is visible in both views

[sidereadframe,sidenframes] = get_readframe_fcn(sidemoviefile);
[frontreadframe,frontnframes] = get_readframe_fcn(frontmoviefile);

%playfmf(sidemoviefile);

% this was set manually while running playfmf
sidebadframes = [1,110
  218,327
  402,514
  528,1607
  1778,1992
  2357,14162];

%playfmf('moviefile',frontmoviefile);

% this was set manually while running playfmf
frontbadframes = [207,491
  600,1524
  1800,2000
  2060,2347
  2394,2889
  2944,3254
  3614,6252
  6500,7085
  7251,8380
  9000,10836
  11000,14128];

sidecansample = true(1,sidenframes);
for i = 1:size(sidebadframes,1),
  sidecansample(sidebadframes(i,1):sidebadframes(i,2)) = false;
end
frontcansample = true(1,frontnframes);
for i = 1:size(frontbadframes,1),
  frontcansample(frontbadframes(i,1):frontbadframes(i,2)) = false;
end

nframes = min(sidenframes,frontnframes);
cansample = sidecansample(1:nframes) & frontcansample(1:nframes);

cansample = find(cansample);
framessample = nan(1,nframessample);
framessample(1) = cansample(1);
mind = abs(framessample(1)-cansample);
for i = 2:nframessample,

  [maxmind,j] = max(mind);
  if maxmind == 0,
    break;
  end
  framessample(i) = cansample(j);
  mind = min(mind,abs(framessample(i)-cansample));
  
end
framessample = unique(framessample(~isnan(framessample)));


%% write images

for i = 1:numel(framessample),
  
  im = sidereadframe(framessample(i));
  imwrite(im,fullfile(outdir,'side',sprintf('side%02d.tif',i)),'tif');
  
  im = frontreadframe(framessample(i));  
  imwrite(im,fullfile(outdir,'front',sprintf('front%02d.tif',i)),'tif');

end

%% calibrate the side view

origdir = pwd;
cd(fullfile(outdir,'side'));
calib_gui;

% results saved to Side_Calib_Results.mat

cd(origdir);

%% calibrate the front view

cd(fullfile(outdir,'front'));
calib_gui;

% results saved to Side_Calib_Results.mat

cd(origdir);

%% stereo calibration

cd(fullfile(outdir,'stereo'));
stereo_gui;

%% small updates to om and T per mouse

%labels = cellfun(@(x,y) [x,'_',y],{rawdata.mouse},{rawdata.session},'Uni',0);
labels = {rawdata.mouse};

[mice,firstmouseidx,mouseidx] = unique(labels);
lambda = .1;

tweakmederr1 = nan(2,numel(mice));
tweakmederr2 = nan(2,numel(mice));
tweakmaderr1 = nan(2,numel(mice));
tweakmaderr2 = nan(2,numel(mice));
tweakT = nan(3,numel(mice));
tweakom = nan(3,numel(mice));

meandT = .0438;
meandom = 0.0366;
meand = (meandT+meandom)/2;

nsamples = 5000;
samples = linspace(0,1,nsamples);

for mousei = 1:numel(mice),
  
  expis = find(mouseidx==mousei);
  
  xL = [cat(1,trxdata(expis).x1)';cat(1,trxdata(expis).y1)'];
  xR = [cat(1,trxdata(expis).x2)'-imsz(2)/2;cat(1,trxdata(expis).y2)'];
  
  dL = sqrt(sum((xL(:,2:end)-xL(:,1:end-1)).^2,1));
  dR = sqrt(sum((xR(:,2:end)-xR(:,1:end-1)).^2,1));
  cdLR = cumsum([0,dL+dR]);
  cdLR = cdLR / cdLR(end);
  j = 0;
  sampleidx = nan(1,nsamples);
  for i = 1:nsamples,
    j0 = j;
    [~,j] = min(abs(cdLR(j0+1:end)-samples(i)));
    j = j + j0;
    sampleidx(i) = j;
  end
  
  xL = xL(:,sampleidx);
  xR = xR(:,sampleidx);
  
  xt = normalize_pixel(xL,fc_left,cc_left,kc_left,alpha_c_left);
  xtt = normalize_pixel(xR,fc_right,cc_right,kc_right,alpha_c_right);
  xt = [xt;ones(1,size(xt,2))];
  xtt = [xtt;ones(1,size(xtt,2))];
  
  [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
  [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
  [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  err1 = mean(sqrt(sum((xL_re - xL).^2,1)));
  err2 = mean(sqrt(sum((xR_re - xR).^2,1)));
  err0 = err1+err2;
  
  lambdacurr = lambda*err0/meand;
  
  [dparams,fval] = fminsearch(@(x) TweakExtrinsicParamsCriterion(x,xL,xR,xt,xtt,om,T,...
    fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right,lambdacurr),...
    zeros(6,1));
  omcurr = om + dparams(1:3);
  Tcurr = T + dparams(4:6);
  [XL,XR] = stereo_triangulation(xL,xR,omcurr,Tcurr,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
  [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
  [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  d1 = median(xL_re-xL,2);
  d2 = median(xR_re-xR,2);
  tweakmederr1(:,mousei) = d1;
  tweakmederr2(:,mousei) = d2;
  s1 = median(abs(bsxfun(@minus,d1,xL_re-xL)),2);
  s2 = median(abs(bsxfun(@minus,d1,xR_re-xR)),2);
  tweakmaderr1(:,mousei) = s1;
  tweakmaderr2(:,mousei) = s2;
  tweakT(:,mousei) = Tcurr;
  tweakom(:,mousei) = omcurr;

  fprintf('Mouse %s: med err1 = %s, err2 = %s, mad err1 = %s, mad err2 = %s\n',...
    mice{mousei},mat2str(d1,3),mat2str(d2,3),mat2str(s1,3),mat2str(s2,3));
  fprintf('dT = %s, dom*180/pi = %s\n',mat2str(dparams(4:6),3),mat2str(dparams(1:3)*180/pi,3));
  
end

%% test initial calibration

expi = find(~isnan([rawdata.auto_GSSS_Chew_0]),1);
t0 = rawdata(expi).auto_GS00_Lift_0;
t1 = rawdata(expi).auto_GSSS_Chew_0;
xL = [trxdata(expi).x1';trxdata(expi).y1'];
xR = [trxdata(expi).x2'-imsz(2)/2;trxdata(expi).y2'];

[XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
[xL_reL] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
[xR_reL] = project_points2(XL,om,T,fc_right,cc_right,kc_right,alpha_c_right);
[xR_reR] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
[xL_reR] = project_points2(bsxfun(@minus,XR,T),-om,zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
fprintf('Max diff in reconstruction of xL from XL and XR: %f\n',max(sum(abs(xL_reL-xL_reR))));
fprintf('Max diff in reconstruction of xR from XL and XR: %f\n',max(sum(abs(xR_reL-xR_reR))));

d1 = mean(xL_reL-xL,2);
d2 = mean(xR_reR-xR,2);

ncolors = 256;
colors = jet(ncolors);
trxlength = 100;

hfig = 1;
figure(hfig);
clf;
hax = createsubplots(1,2,.05);
him = image(readframe(1),'Parent',hax(1));
axis(hax(1),'image');
hold(hax(1),'on');
htruetrx1 = plot(hax(1),nan(1,2),nan(1,2),'-','Color',[0,.8,0],'LineWidth',3);
htruetrx2 = plot(hax(1),nan(1,2),nan(1,2),'-','Color',[0,.8,0],'LineWidth',3);
hretrx1 = plot(hax(1),nan(1,2),nan(1,2),'-','Color',[.8,0,0],'LineWidth',3);
hretrx2 = plot(hax(1),nan(1,2),nan(1,2),'-','Color',[.8,0,0],'LineWidth',3);
htrue1 = plot(hax(1),nan,nan,'o','LineWidth',3,'MarkerSize',12);
htrue2 = plot(hax(1),nan,nan,'o','LineWidth',3,'MarkerSize',12);
hre1 = plot(hax(1),nan,nan,'x','LineWidth',3,'MarkerSize',12);
hre2 = plot(hax(1),nan,nan,'x','LineWidth',3,'MarkerSize',12);
hconn1 = plot(hax(1),nan(2,trxlength),nan(2,trxlength),'c-');
hconn2 = plot(hax(1),nan(2,trxlength),nan(2,trxlength),'c-');

err1 = sqrt(sum((xL_reL - xL).^2,1));
err2 = sqrt(sum((xR_reR - xR).^2,1));
maxerr = max(max(err1),max(err2));
maxz1 = max(XL(3,:));
minz1 = min(XL(3,:));
maxz2 = max(XR(3,:));
minz2 = min(XR(3,:));

minX = min(XL,[],2);
maxX = max(XL,[],2);


h3dtrx = plot3(hax(2),XL(1,:),XL(2,:),XL(3,:),'-','Color','w','LineWidth',3);
hold(hax(2),'on');
set(hax(2),'Color','k');
h3d = plot3(hax(2),nan,nan,nan,'o','LineWidth',3,'MarkerSize',12);
h3dx = plot3(hax(2),nan(1,2),nan(1,2),nan(1,2),':','Color','c');
h3dy = plot3(hax(2),nan(1,2),nan(1,2),nan(1,2),':','Color','m');
h3dz = plot3(hax(2),nan(1,2),nan(1,2),nan(1,2),':','Color','y');
axis(hax(2),'equal');
grid(hax(2),'on');
set(hax(2),'XColor','w','YColor','w','ZColor','w');
xlabel(hax(2),'x');
ylabel(hax(2),'y');
zlabel(hax(2),'z');

% set(hax(2),'CameraPosition',[-11.9566 -19.3823 51.2603],...
% 	'CameraTarget',[-12.278 17.4454 164.609],...
% 	'CameraUpVector',[0 0 1],...
%   'CameraViewAngle',7.86172);

set(hax(2),'CameraPosition',[-4.05762 -60.7666 75.0567],...
	'CameraTarget',[-12.278 17.4454 164.609],...
	'CameraUpVector',[0 0 1],...
  'CameraViewAngle',7.86172);

iconn = 1;

aviobj = VideoWriter('TestReconstruct1.avi');
open(aviobj);

for t = t0:t1,
  
  t0curr = max(t0,t-trxlength+1);
  set(him,'CData',readframe(t));
%   colorierr1 = RescaleToIndex(err1(t),ncolors,0,maxerr);
%   colorierr2 = RescaleToIndex(err2(t),ncolors,0,maxerr);
  colori1 = RescaleToIndex(XL(3,t),ncolors,minz1,maxz1);
  colori2 = RescaleToIndex(XR(3,t),ncolors,minz2,maxz2);
  set(htrue1,'XData',xL(1,t),'YData',xL(2,t),'Color',colors(colori1,:));
  set(htrue2,'XData',xR(1,t)+imsz(2)/2,'YData',xR(2,t),'Color',colors(colori2,:));
  set(hre1,'XData',xL_reL(1,t),'YData',xL_reL(2,t),'Color',colors(colori1,:));
  set(hre2,'XData',xR_reR(1,t)+imsz(2)/2,'YData',xR_reR(2,t),'Color',colors(colori2,:));
  set(hconn1(iconn),'XData',[xL(1,t),xL_reL(1,t)],'YData',[xL(2,t),xL_reL(2,t)],'Color',colors(colori1,:));
  set(hconn2(iconn),'XData',[xR(1,t),xR_reR(1,t)]+imsz(2)/2,'YData',[xR(2,t),xR_reR(2,t)],'Color',colors(colori2,:));
  set(htruetrx1,'XData',xL(1,t0curr:t),'YData',xL(2,t0curr:t));
  set(htruetrx2,'XData',xR(1,t0curr:t)+imsz(2)/2,'YData',xR(2,t0curr:t));
  set(hretrx1,'XData',xL_reL(1,t0curr:t),'YData',xL_reL(2,t0curr:t));
  set(hretrx2,'XData',xR_reR(1,t0curr:t)+imsz(2)/2,'YData',xR_reR(2,t0curr:t));
  iconn = iconn + 1;
  if iconn > trxlength,
    iconn = 1;
  end
  set(h3dtrx,'XData',XL(1,t0curr:t),'YData',XL(2,t0curr:t),'ZData',XL(3,t0curr:t));
  set(h3d,'XData',XL(1,t),'YData',XL(2,t),'ZData',XL(3,t),'Color',colors(colori1,:));
  set(h3dx,'XData',[minX(1),XL(1,t)],'YData',XL(2,t)+[0,0],'ZData',XL(3,t)+[0,0]);
  set(h3dy,'XData',XL(1,t)+[0,0],'YData',[minX(2),XL(2,t)],'ZData',XL(3,t)+[0,0]);
  set(h3dz,'XData',XL(1,t)+[0,0],'YData',XL(2,t)+[0,0],'ZData',[XL(3,t),minX(3)]);
  
  drawnow;
  writeVideo(aviobj,getframe(hfig));
  
end

close(aviobj);

%% look at error of initial calibration over time

mederr1 = nan(2,numel(rawdata));
mederr2 = nan(2,numel(rawdata));
maderr1 = nan(2,numel(rawdata));
maderr2 = nan(2,numel(rawdata));

for expi = 1:numel(rawdata),

%   mousei = mouseidx(expi);
%   Tcurr = tweakT(:,mousei);
%   omcurr = tweakom(:,mousei);
  Tcurr = T;
  omcurr = om;
  
  xL = [trxdata(expi).x1';trxdata(expi).y1'];
  xR = [trxdata(expi).x2'-imsz(2)/2;trxdata(expi).y2'];
  
  [XL,XR] = stereo_triangulation(xL,xR,omcurr,Tcurr,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
  [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
  [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
  
  d1 = median(xL_re-xL,2);
  d2 = median(xR_re-xR,2);
  mederr1(:,expi) = d1;
  mederr2(:,expi) = d2;
  s1 = median(abs(bsxfun(@minus,d1,xL_re-xL)),2);
  s2 = median(abs(bsxfun(@minus,d1,xR_re-xR)),2);
  maderr1(:,expi) = s1;
  maderr2(:,expi) = s2;
  
end

labels = cellfun(@(x,y) [x,'_',y],{rawdata.mouse},{rawdata.session_day},'Uni',0);

[~,order] = sort(labels);

figure(10);
clf;
hax = createsubplots(6,1,[.05,.05;.1,.025]);
axes(hax(1));
patch([1:numel(rawdata),numel(rawdata):-1:1],[mederr1(1,order)-maderr1(1,order),mederr1(1,order(end:-1:1))+maderr1(1,order(end:-1:1))],'r','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(rawdata),mederr1(1,order),'-','Color',[.7,0,0]);
title('x1 error');
axis tight;
axes(hax(2));
patch([1:numel(rawdata),numel(rawdata):-1:1],[mederr1(2,order)-maderr1(2,order),mederr1(2,order(end:-1:1))+maderr1(2,order(end:-1:1))],'m','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(rawdata),mederr1(2,order),'-','Color',[.7,0,.7]);
title('y1 error');
axis tight;
axes(hax(3));
patch([1:numel(rawdata),numel(rawdata):-1:1],[mederr2(1,order)-maderr2(1,order),mederr2(1,order(end:-1:1))+maderr2(1,order(end:-1:1))],'b','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(rawdata),mederr2(1,order),'-','Color',[0,0,.7]);
title('x1 error');
axis tight;
axes(hax(4));
patch([1:numel(rawdata),numel(rawdata):-1:1],[mederr2(2,order)-maderr2(2,order),mederr2(2,order(end:-1:1))+maderr2(2,order(end:-1:1))],'c','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(rawdata),mederr2(2,order),'-','Color',[0,.7,.7]);
title('y1 error');
axis tight;
axes(hax(5));
err1 = sqrt(sum(mederr1.^2,1));
plot(1:numel(rawdata),err1,'.-','Color',[.7,0,.35]);
title('error 1');
axis tight;
axes(hax(6));
err2 = sqrt(sum(mederr2.^2,1));
plot(1:numel(rawdata),err2,'.-','Color',[0,.35,.7]);
title('error 2');

set(hax(1:5),'XTickLabel',{});
xtick = find([true,~strcmp({rawdata(order(1:end-1)).date},{rawdata(order(2:end)).date})]);
% xtick = get(hax(6),'XTick');
% xtick = unique(round(xtick(xtick > 0 & xtick <= numel(rawdata))));
set(hax,'XTick',xtick);

labels = cellfun(@(x,y) [x,'_',y],{rawdata.mouse},{rawdata.date},'Uni',0);

set(hax(6),'XTickLabel',labels(order(xtick)));
rotateticklabel(hax(6));

linkaxes(hax,'x');

%% look at tweaked calibration over time

figure;

clf
[~,order] = sort(mice);

hax = createsubplots(6,1,[.05,.05;.15,.025]);
axes(hax(1));
patch([1:numel(mice),numel(mice):-1:1],[tweakmederr1(1,order)-tweakmaderr1(1,order),tweakmederr1(1,order(end:-1:1))+tweakmaderr1(1,order(end:-1:1))],'r','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(mice),tweakmederr1(1,order),'-','Color',[.7,0,0]);
title('x1 error');
axis tight;
axes(hax(2));
patch([1:numel(mice),numel(mice):-1:1],[tweakmederr1(2,order)-tweakmaderr1(2,order),tweakmederr1(2,order(end:-1:1))+tweakmaderr1(2,order(end:-1:1))],'m','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(mice),tweakmederr1(2,order),'-','Color',[.7,0,.7]);
title('y1 error');
axis tight;
axes(hax(3));
patch([1:numel(mice),numel(mice):-1:1],[tweakmederr2(1,order)-tweakmaderr2(1,order),tweakmederr2(1,order(end:-1:1))+tweakmaderr2(1,order(end:-1:1))],'b','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(mice),tweakmederr2(1,order),'-','Color',[0,0,.7]);
title('x1 error');
axis tight;
axes(hax(4));
patch([1:numel(mice),numel(mice):-1:1],[tweakmederr2(2,order)-tweakmaderr2(2,order),tweakmederr2(2,order(end:-1:1))+tweakmaderr2(2,order(end:-1:1))],'c','FaceAlpha',.2,'LineStyle','none');
hold on;
plot(1:numel(mice),tweakmederr2(2,order),'-','Color',[0,.7,.7]);
title('y1 error');
axis tight;
axes(hax(5));
err1 = sqrt(sum(tweakmederr1.^2,1));
plot(1:numel(mice),err1,'.-','Color',[.7,0,.35]);
title('error 1');
axis tight;
axes(hax(6));
err2 = sqrt(sum(tweakmederr2.^2,1));
plot(1:numel(mice),err2,'.-','Color',[0,.35,.7]);
title('error 2');
set(hax(1:5),'XTickLabel',{});
set(hax,'XTick',1:numel(mice),'Box','off');
set(hax(6),'XTickLabel',mice(order));
rotateticklabel(hax(6));

linkaxes(hax,'x');

figure;
clf;
hax = createsubplots(2,1,[.05,.05;.2,.025]);
axes(hax(1));
plot(1:numel(mice),bsxfun(@minus,tweakT(:,order),T),'.-');
title('dT');
legend('dx','dy','dz');
axes(hax(2));
plot(1:numel(mice),bsxfun(@minus,tweakom(:,order),om)*180/pi,'.-');
title('dom');
legend('dx','dy','dz');

set(hax,'XTick',1:numel(mice),'Box','off','XTickLabel',{});
set(hax(2),'XTickLabel',mice(order));
rotateticklabel(hax(2));

%% set origin

origin = nan(3,numel(mice));
for mousei = 1:numel(mice),
  expis = find(mouseidx==mousei);
  
  xLlift = zeros(2,0);
  xRlift = zeros(2,0);
  for i = 1:numel(expis),
    expi = expis(i);
    t = rawdata(expi).auto_Lift_0;
    if ~isnan(t),
      xLlift(:,end+1) = [trxdata(expi).x1(t);trxdata(expi).y1(t)];
      xRlift(:,end+1) = [trxdata(expi).x2(t)-imsz(2)/2;trxdata(expi).y2(t)];
    end
  end
  
  omcurr = tweakom(:,mousei);
  Tcurr = tweakT(:,mousei);
  [XL] = stereo_triangulation(xLlift,xRlift,omcurr,Tcurr,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
  origin(:,mousei) = median(XL,2);

end

%% save results

tmp = struct;
tmp.om0 = om;
tmp.T0 = T;
tmp.ompermouse = tweakom;
tmp.Tpermouse = tweakT;
tmp.fc_left = fc_left;
tmp.cc_left = cc_left;
tmp.kc_left = kc_left;
tmp.alpha_c_left = alpha_c_left;
tmp.fc_right = fc_right;
tmp.cc_right = cc_right;
tmp.kc_right = kc_right;
tmp.alpha_c_right = alpha_c_right;
tmp.mice = mice;
tmp.origin = origin;

save('CameraCalibrationParams20150217.mat','-struct','tmp');