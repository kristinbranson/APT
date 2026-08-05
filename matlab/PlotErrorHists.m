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
  nplots = nkpts + round(nkpts/10); % a tenth of the space is for the table
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

if isma && ~isempty(fp)
  set(hax(nkpts+1:end,:),'Visible','off');
  delete(hax(nkpts+1:end,:))
  hax = hax(1:nkpts,:);
end

if isempty(kpcolors),
  % make sure we get red
  kpcolors = flipud(hsv((nkpts-1)*5+1));
  kpcolors = kpcolors(1:5:end,:);
end

% The axes/text colors depend only on the theme, not on where the keypoint colors
% came from, and everything below uses them.
if islight,
  axescolor = 'w';
  textcolor = 'k';
else
  axescolor = 'k';
  textcolor = [.99,.99,.99];
end

if isempty(binedges),
  maxerr = prctile(errs(:),maxprctile);
  if ~(isfinite(maxerr) && maxerr > 0)
    % There is no spread in the data to derive bin edges from: either every error
    % is missing (e.g. the tracker predicted nothing for any ground-truth frame,
    % so all rows were dropped as all-NaN) or every error is the same value.
    % Fall back to a unit range so the histograms, empty as they are, still draw
    % rather than histcounts() erroring on NaN or non-increasing bin edges.
    maxerr = 1;
  end
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
    if ncurr == 0,
      % No errors for this keypoint, so plot an empty histogram rather than the
      % NaNs that dividing by a zero count would give.
      frac = zeros(size(counts));
    else
      frac = counts / ncurr;
    end
    y = [zeros(1,nbins);frac(:)';frac(:)'];
    patch(x,[y(:);0],kpcolors(kp,:),'Parent',haxcurr,'EdgeColor',axescolor);
  end
end
set(hax,'XLim',[-1,1]*.01+[binedgesplot(1),binedgesplot(end)],'Color',axescolor,'XColor',textcolor,'YColor',textcolor)
% Match the figure background to the axes background so the tick labels and
% axis labels (drawn in the figure margins in textcolor) stay visible; e.g. in
% the dark theme textcolor is near-white and would be invisible on the default
% light figure background.
if exist('hfig','var')
  set(hfig,'Color',axescolor);
end
linkaxes(hax);
% Keep x tick labels on the last keypoint row (row nkpts) so the Error axis
% values are shown; only strip the rows above it.
set(hax(1:nkpts-1,:),'XTickLabel',{});
% The bulk set above (and linkaxes) can leave the retained bottom row without
% visible tick labels, so force it back to auto to guarantee the error values
% are displayed.
set(hax(nkpts,:),'XTickMode','auto','XTickLabelMode','auto');
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
      s = sprintf('(%d)',kp);
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

table_data = [fp fn];
column_names = {sprintf('False Positives'), sprintf('False Negatives')};
row_names = {'Absolute'};
if ~isempty(ntotal)
  table_data = [table_data; fp/ntotal fn/ntotal];
  row_names = [row_names,{'Fraction'}];
end

% Place the FP/FN table in normalized units (the same coordinate system as the
% plots) within the bottom margin reserved above, below the last keypoint row's
% Error-axis tick labels.  Keeping the table normalized means it scales with the
% figure exactly as the plots do, so the two cannot overlap when the figure is
% resized.  The plots span x in [0.1, 0.9] (left/right border 0.1); the table
% uses the same horizontal extent.
t5 = uitable('Parent', hfig, ...
            'Data', table_data, ...
            'ColumnName', column_names, ...
            'RowName', row_names, ...
            'Units', 'normalized', ...
            'Position', [0.1, 0.02, 0.8, 0.085] ...
            );
% uitable has no relative column widths, so size the data columns from the
% table's current pixel width so they fill it, and keep them updated when the
% figure (and hence the normalized-position table) is resized.  The resize
% callback also fires when the caller resizes the figure right after creation.
set_table_column_widths_(t5);
set(hfig,'SizeChangedFcn',@(s,e)set_table_column_widths_(t5));


function set_table_column_widths_(t)
% Size the uitable data columns so they fill the table's width.  uitable
% column widths are pixel-valued (no relative units), so compute them from the
% table's current pixel width.  Called at creation and on every figure resize.
if ~isvalid(t)
  return
end
oldunits = get(t,'Units');
set(t,'Units','pixels');
pos = get(t,'Position');
set(t,'Units',oldunits);
tablewidthpx = pos(3);
rownamewidthpx = 55;  % approximate width of the row-name column
padpx = 20;           % room for borders and a possible scroll bar
ncols = numel(get(t,'ColumnName'));
availpx = max(tablewidthpx - rownamewidthpx - padpx, ncols*40);
colwidthpx = floor(availpx/ncols);
set(t,'ColumnWidth',repmat({colwidthpx},1,ncols));



