function [img,nr,nc] = getframe_invisible_nocheck(gfdata,sz,removeborder,doresize)

if nargin < 3,
  removeborder = true;
end
if nargin < 4,
  doresize = true;
end

if isfield(gfdata,'grayborder'),
  GRAYBORDER = gfdata.grayborder;
else
  GRAYBORDER = 204;
end

hfig = gfdata.hfig;

img = hardcopy(gfdata.hardcopy_args{:});
set(hfig,'Units',gfdata.units,'Position',gfdata.pos);
if numel(img) == 1,
  fprintf('Could not grab invisible figure. Making visible temporarily.\n');
  set(hfig,'visible','on');
  pause(.1);
  img = hardcopy(gfdata.hardcopy_args{:});
  pause(.1);
  set(hfig,'visible','off');
end

% crop off the extra border
if removeborder,
  isallgray = all(img==GRAYBORDER,3);
end

[nr,nc,tmp] = size(img);
if ~doresize,
  sz(1) = nr; sz(2) = nc;
end

maxfirstcol = nc - sz(2) + 1;
maxfirstrow = nr - sz(1) + 1;
if maxfirstcol <= 0 || maxfirstrow <= 0,
  error('image has size [%d,%d] smaller than input sz = [%d,%d]',nr,nc,sz(1),sz(2));
end
if removeborder,
  firstcol = min(find(~all(isallgray,1),1),maxfirstcol);
  firstrow = min(find(~all(isallgray,2),1),maxfirstrow);
else
  firstcol = 1;
  firstrow = 1;
end
lastcol = firstcol + sz(2) - 1;
lastrow = firstrow + sz(1) - 1;
if removeborder,
  lastcol0 = find(~all(isallgray,1),1,'last');
  lastrow0 = find(~all(isallgray,2),1,'last');
else
  lastcol0 = nc;
  lastrow0 = nr;
end
if lastrow ~= lastrow0 || lastcol ~= lastcol0,
  fprintf('input width = %d, actual width = %d, input height = %d, actual height = %d\n',...
    sz(2),lastcol0-firstcol+1,sz(1),lastrow0-firstrow+1);
end

img = img(firstrow:lastrow,firstcol:lastcol,:);
