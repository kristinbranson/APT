function PlotErrorHists(errs,varargin)

[hpar,kpcolors,prcs,prc_vals,binedges,nbins,maxprctile,kpnames,islight,nperkp,fp,fn,isma,ntotal] = ...
  myparse(varargin,'hparent',[],...
  'kpcolors',[],...
  'prcs',[],'prc_vals',[],...
  'binedges',[],'nbins',50,'maxprctile',98,...
  'kpnames',{},'islight',true,...
  'nperkp',[],'fp',[],'fn',[],'isma',false,'ntotal',[]);

[n,nkpts,nviews] = size(errs);

if isma && ~isempty(fp)
  nplots = nkpts+1;
  yborder = 0.05;
else
  nplots = nkpts;
  yborder = 0.05;
end
if isempty(hpar),
  hfig = figure;
  hax = createsubplots(nplots,nviews,[[.1,.025];[yborder,0.002]],hfig);
  hax = reshape(hax,[nplots,nviews]);
else
  if numel(hpar) == 1 && strcmpi(hpar.Type,'figure'),
    hfig = hpar;
    clf(hfig);
    hax = createsubplots(nplots,nviews,[[.1,.025];[yborder,0.002]],hfig);
    hax = reshape(hax,[nplots,nviews]);
  else
    hax = hpar;
  end
end

if isempty(kpcolors),
  % make sure we get red
  kpcolors = flipud(hsv((nkpts-1)*5+1));
  kpcolors = kpcolors(1:5:end,:);
else
  if islight,
    axescolor = 'w';
    textcolor = 'k';
  else
    axescolor = 'k';
    textcolor = [.99,.99,.99];
  end
end

if isempty(binedges),
  maxerr = prctile(errs(:),maxprctile);
  binedgesplot = linspace(0,maxerr,nbins+1);
  binedges = binedgesplot;
  binedges(end) = inf; % include everything
else
  nbins = numel(binedges)-1;
  binedgesplot = binedges;
  binedges(end) = inf;
end
x = [binedgesplot(1:end-1);binedgesplot(1:end-1);binedgesplot(2:end)];
x = [x(:);binedgesplot(end)];

for viewi = 1:nviews,
  for kp = 1:nkpts,
    haxcurr = hax(kp,viewi);
    counts = histcounts(errs(:,kp,viewi),binedges);
    ncurr = nnz(~isnan(errs(:,kp,viewi)));
    frac = counts / ncurr;
    y = [zeros(1,nbins);frac(:)';frac(:)'];
    patch(x,[y(:);0],kpcolors(kp,:),'Parent',haxcurr,'EdgeColor',axescolor);
  end
end
set(hax,'XLim',[-1,1]*.01+[binedgesplot(1),binedgesplot(end)],'Color',axescolor,'XColor',textcolor,'YColor',textcolor)
linkaxes(hax);
set(hax(1:end-1,:),'XTickLabel',{});
set(hax(:,2:end),'YTickLabel',{});
ylim = get(hax(1),'YLim');

for viewi = 1:nviews,
  for kp = 1:nkpts,
    haxcurr = hax(kp,viewi);
    hold(haxcurr,'on');
    mederr = median(errs(:,kp,viewi));
    plot(haxcurr,[mederr,mederr],[0,ylim(2)],'-','Color',textcolor);
  end
end

for viewi = 1:nviews,
  for kp = 1:nkpts,
    haxcurr = hax(kp,viewi);
    if numel(kpnames) >= kp,
      s = sprintf('(%d) %s',kp,kpnames{kp});
    else
      sprintf('(%d)',kp);
    end
    if nviews > 1,
      s = [s,sprintf(', view %d',viewi)];
    end
    if size(nperkp,1) >= kp && size(nperkp,2) >= viewi && ~isnan(nperkp(kp,viewi)),
      s = [s,sprintf(', n = %d',nperkp(kp,viewi))];
    end
    text(binedgesplot(end),ylim(2),s,'HorizontalAlignment','right',...
      'VerticalAlignment','top','Parent',haxcurr,'Interpreter','none','Color',textcolor);
  end
end
xlabel(hax(nkpts,1),'Error (px)');

if ~isma || isempty(fp)
  % don't add fp/fn table
  return
end

for viewi = 1:nviews,
  for kp = 1:nkpts,
      pp = get(hax(kp,viewi),'Position');
      pp(1) = pp(1) + 0.025;
      set(hax(kp,viewi),'Position',pp);
  end
end

pos = get(hax(end,1), 'Position');
delete(hax(end,1));

table_data = [fp fn];
column_names = {sprintf('False Positives'), sprintf('False Negatives')};
row_names = {'Absolute'};
if ~isempty(ntotal)
  table_data = [table_data; fp/ntotal fn/ntotal];
  row_names = [row_names,{'Fraction'}];
end

t5 = uitable('Parent', hfig, ...
            'Data', table_data, ...
            'ColumnName', column_names, ...
            'RowName', row_names, ...
            'Units', 'normalized', ...
            'Position', [pos(1), 0.025, pos(3), pos(4)+0.01] ...
            );



