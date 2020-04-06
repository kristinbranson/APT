function [vidobj] = MakeMouseTrackingResultsVideo(expdir,trxdata,varargin)

[firstframe,endframe,moviefilestr,resvideo,...
  hfig,vidobj,trxcolor,markercolor,textcolor,maxtrxlen,fps] = ...
  myparse(varargin,'firstframe',1,'endframe',inf,...
  'moviefilestr','movie_comb.avi','resvideo','',...
  'hfig',2,'vidobj',[],'TrxColor',[0,1,1],'MarkerColor',[1,0,1],'TextColor','w','maxtrxlen',500,'fps',30);

[~,n] = fileparts(expdir);

figure(hfig);
clf;
axes('Position',[0,0,1,1]);

readframe = get_readframe_fcn(fullfile(expdir,moviefilestr));
alphas = .99.^(maxtrxlen:-1:1);

nfids = 2; 

p1 = cat(3,cat(2,trxdata.x1,trxdata.y1),cat(2,trxdata.x2,trxdata.y2));

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
htrx = cell(1,nfids);
hcurr = nan(1,nfids);
for fidcurr = 1:nfids,
  htrx{fidcurr} = PlotInterpColorLine([0,0],[0,0],[trxcolor;trxcolor],[0;1],'LineWidth',4);
  hcurr(fidcurr) = plot(nan,nan,'o','Color',markercolor,'LineWidth',5,'MarkerSize',12);
end
htext = text(imsz(2),imsz(1),'0.000s','Color',textcolor,'FontSize',24,...
  'HorizontalAlignment','right','VerticalAlignment','bottom');

set(hfig,'Renderer','OpenGL');
colormap jet;

axis off;

hax = gca;

didopen = false;
if ~isempty(resvideo) && isempty(vidobj),
  didopen = true;
  vidobj = VideoWriter(resvideo);
  vidobj.FrameRate = fps;
  open(vidobj);
end

gfdata = [];
endframe = min(endframe,size(p1,1));

ax = [0,0,0,0];
border = 20;

for f = firstframe:endframe,
  
  im = readframe(f);
  if imsz(3) == 1,
    if isfloat(im) && max(im(:)) > 255,
      im = uint8(im);
    end
    im = repmat(im,[1,1,3]);
  end
  
  set(him,'CData',im);
  
  for fidcurr = 1:nfids,
    xtrx = p1(max(firstframe,f-maxtrxlen):f,1,fidcurr);
    ytrx = p1(max(firstframe,f-maxtrxlen):f,2,fidcurr);
    UpdateInterpColorLine(htrx{fidcurr},'x',xtrx,'y',ytrx,'alphas',alphas(end-numel(xtrx)+1:end),...
      'colors',repmat(trxcolor,[numel(xtrx),1]));
    %set(htrx(fidcurr),'XData',xtrx,'YData',ytrx);
    set(hcurr(fidcurr),'XData',xtrx(end),'YData',ytrx(end));
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

