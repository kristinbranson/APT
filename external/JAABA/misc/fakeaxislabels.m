function fakeaxislabels(real,disp,whichax,hax,formatstr)

if ~exist('whichax','var'),
  whichax = 'x';
end
if ~exist('hax','var') || isempty(hax),
  hax = gca;
end
if ~exist('formatstr','var'),
  formatstr = '%f';
end
if strcmpi(whichax,'y')
  tick = 'ytick';
  ticklabel = 'yticklabel';
else
  tick = 'xtick';
  ticklabel = 'xticklabel';
end

labels = cell(1,length(disp));
for i = 1:length(disp),
  labels{i} = sprintf(formatstr,disp(i));
end

set(hax,tick,real,ticklabel,labels);