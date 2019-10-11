function [ks,oks,okstype,hfig] = ComputeOKS(data,kappa,varargin)

hfig = [];
[distname,pttypes,doplot] = myparse(varargin,'distname','gamma2','pttypes',{},'doplot',false);

[n,npts,~] = size(data.labels);
if isempty(pttypes),
  pttypes = cell(npts,2);
  for i = 1:npts,
    pttypes{i,1} = num2str(i);
    pttypes{i,2} = i;
  end
end
npttypes = size(pttypes,1);

pt2pttypeidx = nan(1,npts);
for i = 1:npttypes,
  pt2pttypeidx(pttypes{i,2}) = i;
end

kappa = kappa(pt2pttypeidx);
errs = sum((data.pred - data.labels).^2,3);
switch distname,
  
  case 'gamma2'
    ks = exp(-errs./(2*kappa));
  case 'gaussian'
    ks = exp(-errs./(2*kappa.^2));
  otherwise
    error('not implemented');
end

oks = mean(ks,2);
okstype = nan(size(ks,1),npttypes);
for pti = 1:npttypes,
  okstype(:,pti) = mean(ks(:,pttypes{pti,2}),2);
end

if doplot,
  
  hfig = figure;
  clf;
  hax = createsubplots(npttypes,1);
  hax = reshape(hax,[npttypes,1]);
  maxerr = prctile(errs(:),99);
%   edges = linspace(0,maxerr,51);
%   ctrs = (edges(1:end-1)+edges(2:end))/2;
  for pti = 1:npttypes,
    colors = lines(numel(pttypes{pti,2}));
    hold(hax(pti,1),'on');

%     counts = histc(vectorize(errs(:,pttypes{pti,2})),edges);
%     frac = counts(1:end-1)/sum(counts(1:end-1));
%     plot(hax(pti,1),ctrs,frac,'-');
%     fracfit = exp(-ctrs./(2*kappa(pti)));
%     fracfit = fracfit / sum(fracfit);
%     plot(hax(pti),ctrs,fracfit,'-');
    for ii = 1:numel(pttypes{pti,2}),
      i = pttypes{pti,2}(ii);
      plot(hax(pti,1),errs(:,i),ks(:,i),'.','Color',colors(ii,:));
    end
    title(hax(pti,1),pttypes{pti,1});
  end
  linkaxes(hax);
  set(hax,'XLim',[0,maxerr]);
end