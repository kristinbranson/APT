function [hax,hfig] = PlotLabeledReachOutliersPerMouse(labeldata,uniquedays,noutliers,varargin)

[hax,hfig,mousename,meanperday,Sperday] = myparse(varargin,'hax',[],'hfig',[],'mousename','',...
  'muperday',{},'Sperday',{});

ndays = numel(uniquedays);
if isempty(hax),
  if isempty(hfig),
    hfig = figure;
  else
    figure(hfig);
  end
  hax = createsubplots(ndays,noutliers,.05,hfig);
  hax = reshape(hax,[ndays,noutliers+1]);
else
  assert(noutliers+1 == size(hax,2));
end

for i = 1:numel(labeldata.labeledpos),
  if ~isempty(labeldata.labeledpos),
    [nlandmarks,nd,~] = size(labeldata.labeledpos{i});
    assert(nd == 2,'Not implemented: can only plot 2d data');
    break;
  end
end

nexps = numel(labeldata.movieFilesAll);
[~,dayidx] = ismember({labeldata.expInfo.datestr},uniquedays);
colors = jet(ndays);

posperday = cell(1,ndays);
expiperday = cell(1,ndays);
tperday = cell(1,ndays);
for dayi = 1:ndays,
  posperday{dayi} = zeros([nlandmarks,nd,0]);
end

for expi = 1:nexps,
  [pos,ts] = SparseLabeledPos(labeldata.labeledpos{expi});
  if isempty(ts),
    continue;
  end
  posperday{dayidx(expi)}(:,:,end+1:end+numel(ts)) = pos;
  expiperday{dayidx(expi)}(end+1:end+numel(ts)) = expi;
  tperday{dayidx(expi)}(end+1:end+numel(ts)) = ts;
end

err = cell(1,ndays);
for dayi = 1:ndays,
  haxcurr = hax(dayi,1);
  mu = median(posperday{dayi},3);
  err{dayi} = squeeze(sum(sqrt(sum((bsxfun(@minus,mu,posperday{dayi})).^2,2)),1));
  if isempty(err{dayi}),
    continue;
  end
  ncurr = numel(err{dayi});
  [sortederr,order] = sort(err{dayi});
  trialnum = [labeldata.expInfo(expiperday{dayi}).trialnum];
  plot(haxcurr,trialnum,err{dayi},'k.');
  hold(haxcurr,'on');
  i0 = max(1,ncurr-noutliers+1);
  noutlierscurr = min(noutliers,numel(err{dayi}));
  plot(haxcurr,trialnum(order(i0:end)),sortederr(i0:end),'ro');
  set(haxcurr,'XLim',[min(trialnum)-1,max(trialnum)+1],'Box','off');
  if dayi == ndays,
    xlabel(haxcurr,'Trial number');
    ylabel(haxcurr,'Distance to median');
  end
  for outlieri = 1:noutlierscurr,
    haxcurr = hax(dayi,outlieri+1);
    expii = order(ncurr-outlieri+1);
    expi = expiperday{dayi}(expii);
    t = tperday{dayi}(expii);
    pos = posperday{dayi}(:,:,expii);
    [readframe] = get_readframe_fcn(labeldata.movieFilesAll{expi});
    im = readframe(t);
    image(im,'Parent',haxcurr);
    axis(haxcurr,'image','off');
    hold(haxcurr,'on');
    if size(meanperday,3) >= dayi,
      meancurr = meanperday(:,:,dayi);
      Scurr = Sperday(:,:,:,dayi);
      if ~any(isnan(meancurr(:))),
        for landmarki = 1:nlandmarks,
          drawcov(meancurr(:,landmarki),Scurr(:,:,landmarki),...
            'Parent',haxcurr,'Color',colors(dayi,:));
        end
      end
    end
    plot(haxcurr,pos(:,1),pos(:,2),'+','Color',colors(dayi,:));
    title(haxcurr,sprintf('%s frame %d',labeldata.expNames{expi},t),'Interpreter','none');
  end
end
maxerr = max(cat(1,err{:}));
set(hax(:,1),'YLim',[0,maxerr*1.01]);
