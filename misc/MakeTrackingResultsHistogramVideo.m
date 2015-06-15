function [vidobj] = MakeTrackingResultsHistogramVideo(expdir,trxfile,varargin)

[firstframe,endframe,moviefilestr,resvideo,smoothsig,maxv,...
  hfig,vidobj,trx_firstframe,intrxfile,fly,winrad,trxcolor,textcolor] = ...
  myparse(varargin,'firstframe',1,'endframe',inf,...
  'moviefilestr','movie_comb.avi','resvideo','',...
  'smoothsig',2,'maxv',1.3,'hfig',2,'vidobj',[],'trx_firstframe',1,...
  'intrxfile','','fly',1,'winrad',100,'TrxColor','m','TextColor','k');

regi = 1;
expi = 1;

[~,n] = fileparts(expdir);

fil = fspecial('gaussian',6*smoothsig+1,smoothsig);

figure(hfig);
clf;
hax = axes('Position',[0,0,1,1]);

readframe = get_readframe_fcn(fullfile(expdir,moviefilestr));

load(trxfile,'phisPrAll','phisPr');

[F,D,K] = size(phisPrAll{regi,expi}); %#ok<USENS>
firstframe = max(firstframe,trx_firstframe);
endframe = min([F,endframe,size(phisPr{1,expi},1)+trx_firstframe-1]);  %#ok<USENS>

istrx = false;
if ~isempty(intrxfile),
  trx = load_tracks(intrxfile);
  istrx = true;
end

d = 2;
nfids = D/d; 

if numel(phisPr) > nfids,
  p1 = cat(3,phisPr{end-nfids+1:end,expi});
else
  p1 = permute(reshape(phisPr{1,expi},[size(phisPr{1,expi},1),nfids,d]),[1,3,2]);
end


im = readframe(1);
imsz = size(im);
if numel(imsz) < 3,
  imsz(3) = 1;
end
if imsz(3) == 1,
  if isfloat(im) && max(im(:)) > 255,
    im = uint8(im);
  end
  im = repmat(im,[1,1,3]);
end
him = image(im); axis image; hold on;
htrx = nan(1,nfids);
for fidcurr = 1:nfids,
  htrx(fidcurr) = plot(nan,nan,'.-','Color',trxcolor,'LineWidth',1);
end
him2 = imagesc(zeros(imsz(1:2)),...
  'AlphaData',ones(imsz(1:2)),'AlphaDataMapping','none',[0,sqrt(maxv)]);
hcurr = nan(1,nfids);
for fidcurr = 1:nfids,
  hcurr(fidcurr) = plot(nan,nan,'+','Color',[0,0,0],'LineWidth',3,'MarkerSize',12);
end
htext = text(imsz(2),imsz(1),'0.000s','Color',textcolor,'FontSize',24,...
  'HorizontalAlignment','right','VerticalAlignment','bottom');

% hcb = colorbar('Location','East');
% set(hcb,'XColor','w','YColor','w');
set(hax,'CLim',[0,maxv]);

set(hfig,'Renderer','OpenGL');
colormap jet;

axis off;

hax = gca;

didopen = false;
if ~isempty(resvideo) && isempty(vidobj),
  didopen = true;
  vidobj = VideoWriter(resvideo);
  open(vidobj);
end

cm = logscale_colormap(jet(512),[0,maxv]);
colormap(cm);

ctrs = {1:imsz(1),1:imsz(2)};

gfdata = [];
firstframe_trx = firstframe-trx_firstframe+1;

ax = [0,0,0,0];
border = 20;

for f = firstframe:endframe,
  
  ftrx = f-trx_firstframe+1;

  if istrx,
    offt = f-trx(fly).firstframe+1;
    xcurr = trx(fly).x(offt) + cos(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
    ycurr = trx(fly).y(offt) + sin(trx(fly).theta(offt))*2*trx(fly).a(offt)*[-1.5,0,1];
    
    if any(xcurr - border < ax(1) | ...
        xcurr + border > ax(2) | ...
        ycurr - border < ax(3) | ...
        ycurr + border > ax(4)),
      
      ax = [xcurr(2)-2*winrad,xcurr(2)+2*winrad,ycurr(2)-2*winrad,ycurr(2)+2*winrad];
      set(hax,'XLim',ax(1:2),'YLim',ax(3:4));
      ax = axis(hax);
      set(htext,'Position',[ax(2)-5,ax(4)-5,0]);
    end
  end
  
  p = reshape(phisPrAll{regi,expi}(ftrx,:,:),[D,K]);
  
  countscurr = 0;
  for fidcurr = 1:nfids,
    countscurr = countscurr + hist3(p([nfids+fidcurr,fidcurr],:)',ctrs);
  end
  density = imfilter(countscurr,fil,'corr','same',0);
  
  im = readframe(f);
  if imsz(3) == 1,
    if isfloat(im) && max(im(:)) > 255,
      im = uint8(im);
    end
    im = repmat(im,[1,1,3]);
  end

  
  set(him,'CData',im);
  set(him2,'CData',density,'AlphaData',min(1,3*sqrt(density)/sqrt(maxv)));

  for fidcurr = 1:nfids,
    xtrx = p1(firstframe_trx:ftrx,1,fidcurr);
    ytrx = p1(firstframe_trx:ftrx,2,fidcurr);
    if istrx,
      badidx = find(xtrx < ax(1) | xtrx > ax(2) | ytrx < ax(3) | ytrx > ax(4),1,'last');
      if ~isempty(badidx),
        xtrx(1:badidx) = nan;
        ytrx(1:badidx) = nan;
      end
    end
    set(htrx(fidcurr),'XData',xtrx,'YData',ytrx);
    set(hcurr(fidcurr),'XData',p1(ftrx,1,fidcurr),'YData',p1(ftrx,2,fidcurr));
  end
  
  set(htext,'String',sprintf('%.3fs',(f-1)/500));
  
  drawnow;
  
  if ~isempty(vidobj),
    
    if isempty(gfdata),
      gfdata = getframe_initialize(hax);
      fr = getframe_invisible(hax);
      gfdata.sz = size(fr);
    end
    fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
    writeVideo(vidobj,fr);
    
  end

end

if didopen,
  close(vidobj);
end

