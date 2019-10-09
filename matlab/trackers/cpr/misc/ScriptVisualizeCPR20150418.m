%% set up paths

addpath ..;
addpath ../video_tracking;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

defaultfolder = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
defaultfile = 'M134labeleddata.mat';

trainresfile_motion_combine = 'TrainedModel_2D_motion_combined_20150324.mat';
moviefilestr = 'movie_comb.avi';

%%

% firstframe = 51;
% endframe = 250;
firstframe = 301;
endframe = 600;
expdir = '/tier2/hantman/Jay/videos/M134C3VGATXChR2/20150303L/CTR/M134_20150303_v014';
testresfile = '';
[phisPr,phisPrAll,phisPrTAll]=test(expdir,trainresfile_motion_combine,testresfile,moviefilestr,[],firstframe,endframe);

%% plot iterations

maxniters = 100;
maxnrestarts = 50;

[~,n] = fileparts(expdir);
resvideo = sprintf('CPRIterations_%s_%dto%d_niters%dnrestarts%d.avi',n,firstframe,endframe,maxniters,maxnrestarts);


hfig = 1;
figure(hfig);
clf;

readframe = get_readframe_fcn(fullfile(expdir,moviefilestr));

im = readframe(1);
imsz = size(im);
hax = axes('Position',[0,0,1,1]);
him = image(im); axis image; hold on;
set(hfig,'Renderer','OpenGL');
colormap jet;

regi = 1;
expi = 1;
T = size(phisPrTAll{regi,expi},4)-1;
K = size(phisPrTAll{regi,expi},2);
F = size(phisPrTAll{regi,expi},1);
D = size(phisPrTAll{regi,expi},3);

set(hax,'CLim',[1,maxniters+1]);
axis off;
hcb = colorbar;
ylabel(hcb,'Iteration')
      
if ~isempty(resvideo),
  vidobj = VideoWriter(resvideo);
  open(vidobj);
end
colors = jet(T+1)*.75+.25;
colors = colors(round(linspace(1,T+1,maxniters+1)),:);
colormap(colors);
truesize;

for f = 1:F,
  
  p_t = reshape(phisPrTAll{regi,expi}(f,1:maxnrestarts,:,1:maxniters+1),[maxnrestarts,D,maxniters+1]);
  p_t = max(.5,p_t);
  p_t(:,[1,2],:) = min(p_t(:,[1,2],:),imsz(2));
  p_t(:,[3,4],:) = min(p_t(:,[3,4],:),imsz(1));
          
  set(him,'CData',readframe(f+firstframe-1));
  delete(findall(hax,'Type','line'));
  delete(findall(hax,'Type','patch'));
  for tmpi = 1:size(p_t,1),
    PlotInterpColorLine(squeeze(p_t(tmpi,1,:)),squeeze(p_t(tmpi,3,:)),colors,.25,'usepatch',false,'LineWidth',1);
  end
  for tmpt = 1:size(p_t,3),
    for tmpi = 1:size(p_t,1),
      plot(squeeze(p_t(tmpi,1,tmpt)),squeeze(p_t(tmpi,3,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6);
    end;
  end
  for tmpi = 1:size(p_t,1),
    PlotInterpColorLine(squeeze(p_t(tmpi,2,:)),squeeze(p_t(tmpi,4,:)),colors,.25,'usepatch',false,'LineWidth',1);
  end
  for tmpt = 1:size(p_t,3),
    for tmpi = 1:size(p_t,1),
      plot(squeeze(p_t(tmpi,2,tmpt)),squeeze(p_t(tmpi,4,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6);
    end;
  end
  
  drawnow;
  
  %disp(size(fr.cdata));
  
  if ~isempty(resvideo),
    
    if f == 1,
      gfdata = getframe_initialize(hax);
      fr = getframe_invisible(hax);
      sz = size(fr);
    end
    fr = getframe_invisible_nocheck(gfdata,sz);
      
    writeVideo(vidobj,fr);
    
  end
          
end

if ~isempty(resvideo),
  close(vidobj);
end

%% histogram distribution of estimates

[~,n] = fileparts(expdir);
resvideo = sprintf('CPRDistribution_%s_%dto%d.avi',n,firstframe,endframe);
%resvideo = '';

smoothsig = 2;
fil = fspecial('gaussian',6*smoothsig+1,smoothsig);
maxv = 1.3;

hfig = 2;
figure(hfig);
clf;
hax = axes('Position',[0,0,1,1]);

readframe = get_readframe_fcn(fullfile(expdir,moviefilestr));

im = readframe(1);
imsz = size(im);
him = image(im); axis image; hold on;
htrx1 = plot(nan,nan,'-','Color',[1,1,1],'LineWidth',3);
htrx1(2) = plot(nan,nan,'--','Color',[.5,0,0],'LineWidth',3);
htrx2 = plot(nan,nan,'-','Color',[1,1,1],'LineWidth',3);
htrx2(2) = plot(nan,nan,'--','Color',[.5,0,0],'LineWidth',3);
him2 = imagesc(zeros(imsz(1:2)),...
  'AlphaData',ones(imsz(1:2)),'AlphaDataMapping','none',[0,sqrt(maxv)]);
hcurr1 = plot(nan,nan,'k+','LineWidth',2,'MarkerSize',8);
hcurr2 = plot(nan,nan,'k+','LineWidth',2,'MarkerSize',8);

% hcb = colorbar('Location','East');
% set(hcb,'XColor','w','YColor','w');
set(hax,'CLim',[0,maxv]);

set(hfig,'Renderer','OpenGL');
colormap jet;

regi = 1;
expi = 1;
[F,D,K] = size(phisPrAll{regi,expi});

axis off;
      
if ~isempty(resvideo),
  vidobj = VideoWriter(resvideo);
  open(vidobj);
end

cm = logscale_colormap(jet(512),[0,maxv]);
colormap(cm);

ctrs = {1:imsz(1),1:imsz(2)};


for f = 1:F,
          
  p = reshape(phisPrAll{regi,expi}(f,:,:),[D,K]);
  
  countscurr = hist3(p([3,1],:)',ctrs) + hist3(p([4,2],:)',ctrs);
  density = imfilter(countscurr,fil,'corr','same',0);
  set(him,'CData',readframe(f+firstframe-1));
  set(him2,'CData',density,'AlphaData',min(1,3*sqrt(density)/sqrt(maxv)));

  set(htrx1,'XData',phisPr{end,expi}(1:f,1),'YData',phisPr{end,expi}(1:f,2));
  set(hcurr1,'XData',phisPr{end,expi}(f,1),'YData',phisPr{end,expi}(f,2));
  set(htrx2,'XData',phisPr{end-1,expi}(1:f,1),'YData',phisPr{end-1,expi}(1:f,2));
  set(hcurr2,'XData',phisPr{end-1,expi}(f,1),'YData',phisPr{end-1,expi}(f,2));
  
  drawnow;
  fr = getframe(hax);
  
  disp(size(fr.cdata));
  
  if ~isempty(resvideo),
            
    writeVideo(vidobj,fr);
    
  end
          
end

if ~isempty(resvideo),
  close(vidobj);
end


%%

hfig = 1;
figure(hfig);
clf; image(repmat(uint8(Is{1}),[1,1,3])); axis image; hold on;
set(hfig,'Renderer','OpenGL');
colormap jet;
set(gca,'CLim',[1,T]);
for tmpi = 1:50, PlotInterpColorLine(squeeze(p_t(tmpi,1,:)),squeeze(p_t(tmpi,3,:)),colors,.25,'LineWidth',3); end
for tmpt = 1:T, for tmpi = 1:50, plot(squeeze(p_t(tmpi,1,tmpt)),squeeze(p_t(tmpi,3,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6); end; end
for tmpi = 1:50, PlotInterpColorLine(squeeze(p_t(tmpi,2,:)),squeeze(p_t(tmpi,4,:)),colors,.25,'LineWidth',3); end
for tmpt = 1:T, for tmpi = 1:50, plot(squeeze(p_t(tmpi,2,tmpt)),squeeze(p_t(tmpi,4,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6); end; end
axis off;
hcb = colorbar;
ylabel(hcb,'Iteration')

savefigdir = '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/Trajectories20150409';

firstframe = 356;
expdir = '/tier2/hantman/Jay/videos/M134C3VGATXChR2/20150303L/CTR/M134_20150303_v014';
[~,n] = fileparts(expdir);
savefig(fullfile(sprintf('CPRIterations_%s_%d.png',n,firstframe)),hfig,'png');
movefile(sprintf('CPRIterations_%s_%d.png',n,firstframe),fullfile(savefigdir,sprintf('CPRIterations_%s_%d.png',n,firstframe)));

%%


hfig = 2;
figure(hfig);
clf; image(repmat(uint8(Is{1}),[1,1,3])); axis image; hold on;
set(hfig,'Renderer','painters');
colormap jet;
set(gca,'CLim',[1,T]);
for tmpi = 1:50, PlotInterpColorLine(squeeze(p_t(tmpi,1,:)),squeeze(p_t(tmpi,3,:)),colors,'usepatch',false,'LineWidth',1); end
for tmpt = 1:T, for tmpi = 1:50, plot(squeeze(p_t(tmpi,1,tmpt)),squeeze(p_t(tmpi,3,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6); end; end
for tmpi = 1:50, PlotInterpColorLine(squeeze(p_t(tmpi,2,:)),squeeze(p_t(tmpi,4,:)),colors,'usepatch',false,'LineWidth',1); end
for tmpt = 1:T, for tmpi = 1:50, plot(squeeze(p_t(tmpi,2,tmpt)),squeeze(p_t(tmpi,4,tmpt)),'o','Color',colors(tmpt,:),'MarkerFaceColor',colors(tmpt,:),'Markersize',6); end; end
axis off;
hcb = colorbar;
ylabel(hcb,'Iteration')

savefigdir = '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/Trajectories20150409';

firstframe = 356;
expdir = '/tier2/hantman/Jay/videos/M134C3VGATXChR2/20150303L/CTR/M134_20150303_v014';
[~,n] = fileparts(expdir);
savefig(fullfile(sprintf('CPRIterations_%s_%d_v2.png',n,firstframe)),hfig,'png');
movefile(sprintf('CPRIterations_%s_%d_v2.png',n,firstframe),fullfile(savefigdir,sprintf('CPRIterations_%s_%d_v2.png',n,firstframe)));
savefig(fullfile(sprintf('CPRIterations_%s_%d_v2.pdf',n,firstframe)),hfig,'pdf');
movefile(sprintf('CPRIterations_%s_%d_v2.pdf',n,firstframe),fullfile(savefigdir,sprintf('CPRIterations_%s_%d_v2.pdf',n,firstframe)));