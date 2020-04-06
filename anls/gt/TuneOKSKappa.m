function [kappa,errs,areas,hfig] = TuneOKSKappa(annoterrdata,varargin)

hfig = [];
[fn,doareanorm,dormoutliers,nremove,pttypes,doplot,distname] = ...
  myparse(varargin,'fieldname','intra','doareanorm',false,'dormoutliers',false,'nremove',0,...
  'pttypes',{},'doplot',false,'distname','gamma2');

npts = size(annoterrdata.(fn){end}.labels,2);
if isempty(pttypes),
  pttypes = cell(npts,2);
  for i = 1:npts,
    pttypes{i,1} = num2str(i);
    pttypes{i,2} = i;
  end
end
npttypes = size(pttypes,1);

areas0 = nan(size(annoterrdata.(fn){end}.labels,1),1);
if doareanorm,
  for j = 1:size(annoterrdata.(fn){end}.labels,1),
    k = convhull(annoterrdata.(fn){end}.labels(j,:,1),annoterrdata.(fn){end}.labels(j,:,2));
    areas0(j) = polyarea(annoterrdata.(fn){end}.labels(j,k,1),annoterrdata.(fn){end}.labels(j,k,2));
  end
else
  areas0(:) = 1;
end
errs0 = sum((annoterrdata.(fn){end}.pred-annoterrdata.(fn){end}.labels).^2,3);
n = size(errs0,1);
errs = cell(1,npttypes);
areas = cell(1,npttypes);
pcompute = [50,75,90,95,97.5];
errprctiles = nan(npttypes,numel(pcompute));
for i = 1:npttypes,
  errprctiles(i,:) = prctile(reshape(errs0(:,pttypes{i,2}),[n*numel(pttypes{i,2}),1]),pcompute);
  errs{i} = reshape(errs0(:,pttypes{i,2})./areas0,[n*numel(pttypes{i,2}),1]);
  areas{i} = repmat(areas0,[numel(pttypes{i,2}),1]);
  fprintf('%s:',pttypes{i,1});
  for j = 1:numel(pcompute),
    fprintf(' %.1f->%.2f',pcompute(j),errprctiles(i,j));
  end
  fprintf('\n');
end

noutliers = zeros(1,npttypes);
if dormoutliers || nremove > 0,
  errprctiles_rmoutliers = nan(npttypes,numel(pcompute));
  for i = 1:npttypes,
    [sortederrs,order] = sort(errs{i});
    if dormoutliers,
%       idxremove = isoutlier(errs{i});
      derr = diff(sortederrs);
      j = find(derr > 1,1);
      idxremove = order(j:end);
    else
      idxremove = order(end-nremove+1:end);
    end
    noutliers(i) = nnz(idxremove);
    errs{i}(idxremove,:) = [];
    areas{i}(idxremove) = [];
    errprctiles_rmoutliers(i,:) = prctile(errs{i},pcompute);
    fprintf('rmoutliers %s:',pttypes{i,1});
    for j = 1:numel(pcompute),
      fprintf(' %.1f->%.2f',pcompute(j),errprctiles_rmoutliers(i,j));
    end
    fprintf('\n');
  end
end

%mederr = nan(1,npttypes);
kappa = nan(1,npttypes);
for i = 1:npttypes,
  errcurr = errs{i};
  switch distname,
    case 'gamma2'
      mederr = median(errcurr,1);
      sigstry = linspace(mederr/10,mederr*10,10000);
      kappa(i) = fitgamma2(errcurr,sigstry);
    case 'gaussian'
      kappa(i) = sqrt(mean(errcurr(:)))*2;
    otherwise
      error('not implemented');
  end
end
%kappa = mederr;

if doplot,
  hfig = figure;
  clf;
  hax = createsubplots(npttypes,1,[.05,.025;.05,.025]);
  if dormoutliers,
    maxerr = prctile(cat(1,errs{:}),100);
  else
    maxerr = prctile(cat(1,errs{:}),99);
  end

  for pti = 1:npttypes,
    
    %fracfit = 1./(2*kappa(pti)) .* exp(-ctrs(:)./(2*kappa(pti)));
    switch distname,
      case 'gamma2',
        edges = linspace(0,maxerr,51);
        ctrs = (edges(1:end-1)+edges(2:end))/2;
        fracfit = exp(-ctrs(:)./(2*kappa(pti)));
        issquarederr = true;
        counts = histc(errs{pti},edges);

      case 'gaussian'
        edges = linspace(0,2*sqrt(maxerr),51);
        ctrs = (edges(1:end-1)+edges(2:end))/2;
        fracfit = exp(-ctrs(:).^2./(2*kappa(pti)^2));
        issquarederr = false;
        counts = histc(sqrt(errs{pti}),edges);
      otherwise
        error('not implemented');
    end
    
    fracfit = fracfit ./ sum(fracfit,1);
    frac = counts(1:end-1)./sum(counts(1:end-1));
    plot(hax(pti),ctrs(:),frac,'k-','LineWidth',2);
    hold(hax(pti),'on');
    plot(hax(pti),ctrs(:),fracfit,'r:','LineWidth',2);
    title(hax(pti),pttypes{pti});
    axisalmosttight([],hax(pti));
    box(hax(pti),'off');
  end
  linkaxes(hax);
  if issquarederr,
    xlabel(hax(end),'Sum-squared error (px)');
  else
    xlabel(hax(end),'Error (px)');
  end
end

%%
function [sigbest,maxll] = fitgamma2(errs,sigstry)

maxll = -inf;
sigbest = nan;
for i = 1:numel(sigstry),
  sig = sigstry(i);
  ll = sum(-log(2*sig) - errs./(2*sig),1);
  if ll > maxll,
    maxll = ll;
    sigbest = sig;
  end
end

