% Usage:
% h = createsubplots(nrows,ncols,border)
% h = createsubplots(nrows,ncols,[borderx,bordery])
% h = createsubplots(nrows,ncols,[[sideborderx, middleborderx];[sidebordery, middlebordery]])
% h = createsubplots(...,[hfig]);
function h = createsubplots(n1,n2,border,hfig)

if nargin == 0,
  error('usage: createsubplots(n1,[n2],[border])');
elseif nargin == 1,
  n = n1;
  n2 = ceil(sqrt(n));
  n1 = ceil(n / n2);
  border = [.02,.02];
elseif nargin == 2,
  if length(n2) == 1 || n2 >= 1,
    n = n1*n2;
    border = [.02,.02];
  else
    border = n2;
    n = n1;
    n2 = ceil(sqrt(n));
    n1 = ceil(n/n2);
  end
else
  n = n1*n2;
end;

if numel(border) == 1,
  border = repmat(border,[2,2]);
elseif numel(border) == 2,
  border = repmat(border(:),[1,2]);
else
  border = reshape(border,[2,2]);
end

if verLessThan('matlab','8.4.0'),
  h = zeros(1,n1*n2);
end
if ~exist('hfig','var'),
  hfig = clf;
elseif ~ishandle(hfig),
  figure(hfig);
else
  clf(hfig);
end
width = (1 - 2*border(1,1) - (n2-1)*border(1,2))/n2;
height = (1 - 2*border(2,1) - (n1-1)*border(2,2))/n1;
left = border(1,1);
for c = 1:n2,
  bottom = 1 - border(2,1) - height;
  for r = 1:n1,
    i = sub2ind([n1,n2],r,c);
    h(i) = axes('position',[left,bottom,width,height],'parent',hfig);
    bottom = bottom - border(2,2) - height;
  end
  left = left + border(1,2) + width;
end;