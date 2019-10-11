function makeResultsMovie(movfile,movout,varargin)
% Create tracking/label results movie
%
% movfile: full path, input movie
%
% movout: full path, output movie

[trkfile,framerate,gamma] = myparse(varargin,...
  'trkfile',[],... % full path to trkfile holding labels
  'framerate',24,... 
  'gamma',[]);

% input mov
mr = MovieReader();
mr.open(movfile);
mr.forceGrayscale = true;

% input lbls
assert(~isempty(trkfile),'No trkfile specified.'); % currently trkfile is only way to input labels
trk = load(trkfile,'-mat');
nptTrk = size(trk.pTrk,1);
nfrm = size(trk.pTrk,3);
trkFrms = arrayfun(@(x)nnz(isnan(trk.pTrk(:,:,x)))==0,1:nfrm);
frmsMov = find(trkFrms);

% colormap
cmap = gray(256);
if ~isempty(gamma)
  cmap = imadjust(cmap,[],[],gamma);
end

% colors
ptClrs = hsv(nptTrk);

if exist(movout,'file')>0
  error('makeResultsMovie:mov','Movie ''%s'' already exists.',movout);
end


hFig = figure;
ax = axes;
im0 = uint8(zeros(mr.nr,mr.nc));
hIm = imagesc(im0,'parent',ax);
colormap(ax,cmap);
truesize(hFig);
hold(ax,'on');
ax.XTick = [];
ax.YTick = [];

hLine = gobjects(nptTrk,1);
for ipt = 1:nptTrk
  hLine(ipt) = plot(ax,nan,nan,'.',...
    'markersize',28,...
    'Color',ptClrs(ipt,:));
end

vr = VideoWriter(movout);
vr.FrameRate = framerate;
vr.open();

hTxt = text(10,15,'','parent',ax,'Color','white','fontsize',24);
hWB = waitbar(0,'Writing video');

for iF=1:numel(frmsMov)
  f = frmsMov(iF);
  
  im = mr.readframe(f);
  hIm.CData = im;
  
  xyf = trk.pTrk(:,:,f);
  for ipt=1:nptTrk
    hLine(ipt).XData = xyf(ipt,1);
    hLine(ipt).YData = xyf(ipt,2);
  end
  
  hTxt.String = sprintf('%04d',f);
  drawnow;
  
  tmpFrame = getframe(ax);
  vr.writeVideo(tmpFrame);
  waitbar(iF/numel(frmsMov),hWB,sprintf('Wrote frame %d\n',f));
end
       
vr.close();
delete(hTxt);
delete(hWB);