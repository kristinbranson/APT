function hfigs = PlotExampleLabelsAndPreds(lbld,predfns,movieidx,frames,varargin)

[hfigbase,naxperfig,landmarkcolors,figpos,prednames,err] = ...
  myparse(varargin,'hfigbase',100,'naxperfig',5,...
  'landmarkcolors',[],'figpos',[10,10,2100,1500],...
  'prednames',{},'err',[]);

nviews = size(lbld.movieFilesAll,2);
nlandmarks = size(lbld.labeledpos{1},1)/nviews;
npredfns = numel(predfns);

if isempty(landmarkcolors),
  landmarkcolors = lines(nlandmarks);
end

if isempty(prednames),
  prednames = predfns;
end

expiprev = nan;
nsamples = numel(frames);
assert(nsamples == numel(movieidx));
nfigs = ceil(nsamples/naxperfig);
hfigs = nan(1,nfigs);

for figi = 1:nfigs,
  i0 = (figi-1)*naxperfig + 1;
  i1 = min((figi-1)*naxperfig + naxperfig,numel(frames));
  hfig = hfigbase + figi;
  hfigs(figi) = hfig;
  figure(hfig);
  clf;
  set(hfig,'Units','pixels','Position',figpos);
  naxcurr = i1-i0+1;
  hax = createsubplots((npredfns+1),nviews*naxcurr,0);
  hax = reshape(hax,[(npredfns+1),nviews,naxcurr]);
  for ii = 1:naxcurr,
    expi = movieidx(i0+ii-1);
    f = frames(i0+ii-1);
    id = lbld.animalids(expi);
    movienum = find(find(lbld.animalids==id)==expi);
    if expi ~= expiprev,
      expiprev = expi;
      readframes = cell(1,nviews);
      for v = 1:nviews,
        readframes{v} = get_readframe_fcn(lbld.movieFilesAll{expi,v});
      end
    end
    for v = 1:nviews,
      im = readframes{v}(f);
      if size(im,3) == 1,
        im = repmat(im,[1,1,3]);
      end
      for k = 1:npredfns+1,
        %axi = sub2ind([npredfns+1,nviews,npredfns+1],k,v,k);
        image(hax(k,v,ii),im);
        axis(hax(k,v,ii),'image','off');
        hold(hax(k,v,ii),'on');
        if k == 1,
          predfn = 'labeledpos';
        else
          predfn = predfns{k-1};
        end
        for l = 1:nlandmarks,
          plot(hax(k,v,ii),lbld.(predfn){expi}((v-1)*nlandmarks+l,1,f),lbld.(predfn){expi}((v-1)*nlandmarks+l,2,f),...
            '+','Color',landmarkcolors(l,:),'MarkerSize',8,'LineWidth',2);
        end
        if k == 1,
          predname = 'Manual';
        else
          predname = prednames{k-1};
        end
        if ii == 1,
          text(5,size(im,1)-5,sprintf('%s, %s',predname,lbld.cfg.ViewNames{v}),...
            'HorizontalAlignment','left','VerticalAlignment','bottom',...
            'color',[.99,.99,.99],'parent',hax(k,v,ii));
        end
        ht = text(5,5,sprintf('animal %d, movie %d, frame %d',id,movienum,f),'Color',[.99,.99,.99],...
          'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(k,v,ii));
        if numel(err) >= ii && k > 1,
          pos = get(ht,'Extent');
          text(5,ceil(pos(2)),sprintf('Total err = %f px',err(k-1,ii)),'Color',[.99,.99,.99],...
            'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(k,v,ii));
        end
      end
    end
    drawnow;
  end
%   if true,
%     set(hfig,'Color','w','InvertHardCopy','off');
%     SaveFigLotsOfWays(hfig,sprintf('TrackingExamples_%02d_%s',figi,figsavestr),{'pdf','fig','png'});
%   end

  
end