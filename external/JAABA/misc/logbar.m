function hh = logbar(edges,vals,width,colors,hax)

if nargin < 3,
  width = .8;
end

edges = edges(:)';
nbins = numel(edges)-1;
sz = size(vals);
if ~any(sz==nbins),
  error('vals should be ndata x nbins');
end

if ~ismatrix(vals),
  error('vals must be 1- or 2-dimensional');
end
if sz(2) ~= nbins && sz(1) == nbins,
  vals = vals';
end
ndata = size(vals,1);

if nargin < 5,
  hax = gca;
end
hfig = get(hax,'Parent');


if nargin < 4 || isempty(colors),
  colors = get(hfig,'Colormap');
  if size(colors,1) > ndata,
    colors = colors(round(linspace(1,size(colors,1),ndata)),:);
  elseif size(colors,1) < ndata,
    colors = colors(modrange(1:ndata,1,size(colors,1)),:);
  end
elseif size(colors,1) < ndata,
  colors = colors(modrange(1:ndata,1,size(colors,1)),:);
end

dedge = diff(edges)*width;
off0 = (1-width)/2;
hh = nan(1,ndata);
holdstate = ishold;
for i = 1:ndata,
  off1 = off0 + (i-1)/ndata;
  off2 = off0 + i/ndata;
  x1 = edges(1:end-1)+off1*dedge;
  x2 = edges(1:end-1)+off2*dedge;
  x = [x1;x1;x2;x2];
  y = [zeros(1,nbins);vals([i,i],:);zeros(1,nbins)];
  hh(i) = patch(x(:),y(:),colors(i,:),'Parent',hax);
  if i == 1,
    hold(hax,'on');
  end
end
if holdstate == 0,
  hold(hax,'off');
end

set(hax,'XScale','log');