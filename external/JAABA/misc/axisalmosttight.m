% axisalmosttight(border,varargin)

function axisalmosttight(border,varargin)

if nargin < 1 || isempty(border),
  border = 1/20;
end
if numel(varargin)>=1,
  hax = varargin{1};
else
  hax = gca;
end

ax = [inf,-inf,inf,-inf];
for i = 1:numel(hax),
  axis(hax(i),'tight');
  ax1 = axis(hax);
  nd = numel(ax1)/2;
  if i == 1,
    ax = repmat([inf,-inf],[1,nd]);
  end
  ax(1:2:end-1) = min(ax(1:2:end-1),ax1(1:2:end-1));
  ax(2:2:end) = max(ax(2:2:end),ax1(2:2:end));
end

if strcmp(hax(1).XScale,'log'),
  ax(1:2) = log10(ax(1:2));
end
if strcmp(hax(1).YScale,'log'),
  ax(3:4) = log10(ax(3:4));
end
dx = (ax(2:2:end)-ax(1:2:end-1))*border;
ax(1:2:end-1) = ax(1:2:end-1) - dx;
ax(2:2:end) = ax(2:2:end) + dx;
if strcmp(hax(1).XScale,'log'),
  ax(1:2) = 10.^(ax(1:2));
end
if strcmp(hax(1).YScale,'log'),
  ax(3:4) = 10.^(ax(3:4));
end
for i = 1:numel(hax),
  axis(hax(i),ax);
end