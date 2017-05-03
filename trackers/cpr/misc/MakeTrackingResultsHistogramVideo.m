function [vidobj] = MakeTrackingResultsHistogramVideo(expdirs,trxfile,varargin)

[firstframe,endframe,moviefilestr,resvideo,smoothsig,maxv,...
  hfig,vidobj,trx_firstframe,intrxfile,fly,winrad,trxcolor,textcolor,...
  allpredfn,predfn,lkcolors,plottrxlen,figpos,plotdensity,...
  CurrMarkerSize,CurrLineWidth,visible,cropframelims] = ...
  myparse(varargin,'firstframe',1,'endframe',inf,...
  'moviefilestr','movie_comb.avi','resvideo','',...
  'smoothsig',2,'maxv',1.3,'hfig',2,'vidobj',[],'trx_firstframe',1,...
  'intrxfile','','fly',1,'winrad',100,'TrxColor','m','TextColor','k',...
  'allpredfn','cpr_all2d_locs','predfn','cpr_2d_locs','lkcolors','k','plottrxlen',inf,...
  'figpos',[],'plotdensity',true,'CurrMarkerSize',12,'CurrLineWidth',3,'Visible','',...
  'cropframelims',[]);

regi = 1;
v = 1;

if isempty(visible),
  if isempty(resvideo),
    visible = 'on';
  else
    visible = 'off';
  end
end

if plotdensity,
  fil = fspecial('gaussian',6*smoothsig+1,smoothsig);
end

vars = whos('-file',trxfile);
if ismember('phisPrAll',{vars.name}),
  if plotdensity,
    load(trxfile,'phisPrAll','phisPr');
  else
    load(trxfile,'phisPr');
  end
else
  assert(ismember('R',{vars.name}));
  load(trxfile,'R');
  nviews = numel(R);
  if plotdensity,
    phisPrAll = cell(1,nviews);
  end
  phisPr = cell(1,nviews);
  isexpdir = ~isempty(expdirs);
  if ~isexpdir,
    expdirs = cell(1,nviews);
  end
  for v = 1:nviews,
    if plotdensity,
      phisPrAll{v} = ConvertMayanklocs2CPRphisPr(R{v}.(allpredfn));
    end
    phisPr{v} = ConvertMayanklocs2CPRphisPr(R{v}.(predfn));
    if ~isexpdir,
      expdirs{v} = R{v}.movie;
    end
  end
  
end


% expdir will hold a movie or multiple movies
if ~iscell(expdirs),
  expdirs = {expdirs};
end
nviews = numel(expdirs);

readframes = cell(1,nviews);
for v = 1:nviews,
  if isempty(moviefilestr),
    readframes{v} = get_readframe_fcn(expdirs{v});
  else
    readframes{v} = get_readframe_fcn(fullfile(expdirs{v},moviefilestr));
  end
end

[F,D] = size(phisPr{1,v});
if plotdensity,
  K = size(phisPrAll{regi,v},3);
end
firstframe = max(firstframe,trx_firstframe);
endframe = min([F,endframe,size(phisPr{1,v},1)+trx_firstframe-1]); 

istrx = false;
if ~isempty(intrxfile),
  if nviews > 1,
    error('multiple views with intrxfile not implemented');
  end
  trx = load_tracks(intrxfile);
  istrx = true;
end

d = 2;
nfids = D/d; 

p1 = cell(1,nviews);
for v = 1:nviews,
  if size(phisPr,1) > nfids,
    p1{v} = cat(3,phisPr{end-nfids+1:end,v});
  else
    p1{v} = permute(reshape(phisPr{1,v},[size(phisPr{1,v},1),nfids,d]),[1,3,2]);
  end
end

islkcolors = size(lkcolors,1)>= nfids;
if ~islkcolors,
  lkcolors = [0,0,0];
end

figure(hfig);
clf;
if ~isempty(figpos),
  set(hfig,'Units','pixels','Position',figpos);
end
if nviews == 1,
  hax = axes('Position',[0,0,1,1]);
else
  hax = createsubplots(1,nviews,0);
end
him = nan(1,nviews);
if islkcolors,
  him2 = nan(nfids,nviews);
else
  him2 = nan(1,nviews);
end
htrx = nan(nfids,nviews);
hcurr = nan(nfids,nviews);
imsz = ones(nviews,3);
ctrs = cell(nviews,2);
for v = 1:nviews,
  im = readframes{v}(1);
  
  if size(cropframelims,1) >= v && ~isnan(cropframelims(v,1)),
    im = im(cropframelims(v,3):cropframelims(v,4),cropframelims(v,1):cropframelims(v,2),:);
    xlim = cropframelims(v,1:2);
    ylim = cropframelims(v,3:4);
  else
    xlim = [1,size(im,2)];
    ylim = [1,size(im,1)];
  end
  
  imszcurr = size(im);
  imsz(v,1:numel(imszcurr)) = imszcurr;
  if imsz(v,3) == 1,
    if isfloat(im) && max(im(:)) > 255,
      im = uint8(im);
    end
    im = repmat(im,[1,1,3]);
  end
  him(v) = image(xlim,ylim,im,'Parent',hax(v)); axis(hax(v),'image','off'); hold(hax(v),'on');
  
  for fidcurr = 1:nfids,
    htrx(fidcurr,v) = plot(hax(v),nan,nan,'.-','Color',trxcolor,'LineWidth',1);
  end
  if plotdensity,
    if islkcolors,
      for l = 1:nfids,
        him2(l,v) = image(xlim,ylim,repmat(reshape(lkcolors(l,:),[1,1,3]),imsz(v,1:2)),...
          'AlphaData',ones(imsz(v,1:2)),'AlphaDataMapping','scaled',...
          'Parent',hax(v));
      end
    else
      him2(v) = imagesc(xlim,ylim,zeros(imsz(v,1:2)),...
        'AlphaData',ones(imsz(v,1:2)),'AlphaDataMapping','none',...
        'Parent',hax(v),[0,sqrt(maxv)]);
    end
  end
  for fidcurr = 1:nfids,
    colorcurr = lkcolors(min(fidcurr,size(lkcolors,1)),:);
    if numel(colorcurr) == 3,
      colorcurr = colorcurr*.6;
    end
    hcurr(fidcurr,v) = plot(hax(v),nan,nan,'+','Color',colorcurr,'LineWidth',CurrLineWidth,'MarkerSize',CurrMarkerSize);
  end
  
  ctrs(v,:) = {ylim(1):ylim(2),xlim(1):xlim(2)};  

end

v = 1;
htext = text(imsz(v,2),imsz(v,1),'0.000s','Color',textcolor,'FontSize',24,...
  'HorizontalAlignment','right','VerticalAlignment','bottom','Parent',hax(v));
set(hfig,'Visible',visible);
drawnow;

% hcb = colorbar('Location','East');
% set(hcb,'XColor','w','YColor','w');
set(hax,'CLim',[0,maxv]);

set(hfig,'Renderer','OpenGL');
colormap jet;

axis off;

didopen = false;
if ~isempty(resvideo) && isempty(vidobj),
  didopen = true;
  vidobj = VideoWriter(resvideo);
  open(vidobj);
end

if ~islkcolors,
  cm = logscale_colormap(jet(512),[0,maxv]);
  colormap(cm);
end


gfdata = [];
firstframe_trx = firstframe-trx_firstframe+1;

ax = [0,0,0,0];
border = 20;

if nviews == 1,
  hvid = hax;
else
  hvid = hfig;
end

for f = firstframe:endframe,
  
  fprintf('Frame %d / %d\n',f-firstframe+1,endframe-firstframe+1);
  
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

  for v = 1:nviews,

    if plotdensity,
      p = reshape(phisPrAll{regi,v}(ftrx,:,:),[D,K]);
      if islkcolors,
        for fidcurr = 1:nfids,
          counts = hist3(p([nfids+fidcurr,fidcurr],:)',ctrs(v,:));
          density = min(imfilter(counts,fil,'corr','same',0),maxv);
          set(him2(fidcurr,v),'AlphaData',min(1,3*sqrt(density)/sqrt(maxv)));
        end
      else
        counts = 0;
        for fidcurr = 1:nfids,
          counts = counts + hist3(p([nfids+fidcurr,fidcurr],:)',ctrs(v,:));
        end
        density = imfilter(counts,fil,'corr','same',0);
        set(him2(v),'CData',density,'AlphaData',min(1,3*sqrt(density)/sqrt(maxv)));
      end
    end
    
    im = readframes{v}(f);
    if size(cropframelims,1) >= v && ~isnan(cropframelims(v,1)),
      im = im(cropframelims(v,3):cropframelims(v,4),cropframelims(v,1):cropframelims(v,2),:);
    end
    if imsz(v,3) == 1,
      if isfloat(im) && max(im(:)) > 255,
        im = uint8(im);
      end
      im = repmat(im,[1,1,3]);
    end

  
    set(him(v),'CData',im);

    f0 = max(ftrx-plottrxlen+1,firstframe_trx);
    for fidcurr = 1:nfids,
      xtrx = p1{v}(f0:ftrx,1,fidcurr);
      ytrx = p1{v}(f0:ftrx,2,fidcurr);
      if istrx,
        badidx = find(xtrx < ax(1) | xtrx > ax(2) | ytrx < ax(3) | ytrx > ax(4),1,'last');
        if ~isempty(badidx),
          xtrx(1:badidx) = nan;
          ytrx(1:badidx) = nan;
        end
      end
      set(htrx(fidcurr,v),'XData',xtrx,'YData',ytrx);
      set(hcurr(fidcurr,v),'XData',p1{v}(ftrx,1,fidcurr),'YData',p1{v}(ftrx,2,fidcurr));
    end
  end
  
  set(htext,'String',sprintf('%.3fs',(f-1)/500));
  
  drawnow;
  
  if ~isempty(vidobj),
    
    if isempty(gfdata),
      gfdata = getframe_initialize(hvid);
      fr = getframe_invisible(hvid);
      gfdata.sz = size(fr);
    end
    fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
    writeVideo(vidobj,fr);
    
  end

end

if didopen,
  close(vidobj);
end

