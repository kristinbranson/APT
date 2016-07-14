function [uniquedays,muperday,Sperday,nperday,hax,hfig] = PlotLabeledReachesPerMouse(labeldata,varargin)

[hax,hfig,mousename,sepdays] = myparse(varargin,'hax',[],'hfig',[],'mousename','','sepdays',true);

if isempty(hax),
  if isempty(hfig),
    hfig = figure;
  else
    figure(hfig);
  end
  hax = gca;
end

for i = 1:numel(labeldata.labeledpos),
  if ~isempty(labeldata.labeledpos),
    [nlandmarks,nd,~] = size(labeldata.labeledpos{i});
    assert(nd == 2,'Not implemented: can only plot 2d data');
    break;
  end
end

assert(exist('nd','var')>0);

isfirst = true;
nexps = numel(labeldata.movieFilesAll);
if sepdays,
  [uniquedays,~,dayidx] = unique({labeldata.expInfo.datestr});
  ndays = numel(uniquedays);
  colors = jet(ndays);
  colorspertrial = colors*.8;
else
  uniquedays = {''};
  dayidx = ones(1,numel(labeldata.expInfo));
  ndays = 1;
  colors = [.8,0,0];
  colorspertrial = [1,1,1];
end

hdays = nan(1,ndays);
muperday = zeros([nd,nlandmarks,ndays]);
Sperday = zeros([nd,nd,nlandmarks,ndays]);
nperday = zeros(1,ndays);

for expi = 1:nexps,
  [pos,ts] = SparseLabeledPos(labeldata.labeledpos{expi});
  if isempty(ts),
    continue;
  end
  if isfirst,
    
    moviefile = labeldata.movieFilesAll{expi};
    if ~exist(moviefile,'file'),
      warning('Video %s does not exist',moviefile);
      im = zeros([labeldata.movieInfoAll{1}.info.nr,labeldata.movieInfoAll{1}.info.nc,3]);
    else
      [readframe] = get_readframe_fcn(moviefile);
      im = readframe(ts(1));
    end
    image(im,'Parent',hax);
    axis(hax,'image');
    hold(hax,'on');
    
    isfirst = false;
  end
  
  if numel(ts) > 1,
    warning('%s: %d labeled frames\n',labeldata.expNames{expi},numel(ts));
  end
  hdays(dayidx(expi)) = plot(hax,vectorize(pos(:,1,:)),vectorize(pos(:,2,:)),'.',...
    'Color',colorspertrial(dayidx(expi),:));
  
  nperday(dayidx(expi)) = nperday(dayidx(expi)) + numel(ts);
  muperday(:,:,dayidx(expi)) = muperday(:,:,dayidx(expi)) + sum(pos,3)';
  for landmarki = 1:nlandmarks,
    for ti = 1:numel(ts),
      Sperday(:,:,landmarki,dayidx(expi)) = Sperday(:,:,landmarki,dayidx(expi)) + ...
        pos(landmarki,:,ti)'*pos(landmarki,:,ti);
    end
  end
  
  drawnow;
  
end

muperday = bsxfun(@rdivide,muperday,reshape(nperday,[1,1,ndays]));
for landmarki = 1:nlandmarks,
  for dayi = 1:ndays,
    Sperday(:,:,landmarki,dayi) = Sperday(:,:,landmarki,dayi)/nperday(dayi) - ...
      muperday(:,landmarki,dayi)*muperday(:,landmarki,dayi)';
  end
end

for dayi = 1:ndays,
  plot(hax,muperday(1,:,dayi),muperday(2,:,dayi),'ks','MarkerFaceColor',colors(dayi,:),'MarkerSize',10);
end
for landmarki = 1:nlandmarks,
  for dayi = 1:ndays,
    htmp = drawcov(muperday(:,landmarki,dayi)',Sperday(:,:,landmarki,dayi),...
      'Parent',hax,'Color',colors(dayi,:),'LineWidth',2);
  end
end

if sepdays,
  hleg = legend(hax,hdays(~isnan(hdays)),uniquedays(~isnan(hdays)));
  set(hleg,'Color','none','EdgeColor','w','TextColor','w')
end

if ~isempty(mousename),
  title(hax,mousename);
end

if isfirst,
  error('No data for mouse %s',mousename);
end