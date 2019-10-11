function iconRGB = makeToolbarIconFromGIF(filename)

% Copyright 2003 The MathWorks, Inc.

% Icon's background color is [0 1 1]
[x,map] = imread(filename);
iconRGB = ind2rgb(x,map);

idx = 1; % fake it to get selecticon to work.

% The GIFs we have seem to have the transparent pixels as an index
% into the last non [0 0 0] entry in the colormap which has the color
% [1 1 1]. Find this index and make these pixels transparent, using
% NaNs.
%
% Some also have the [1 1 1] entry as the last entry in the
% colormap. (helpicon.gif
for i=1:size(map,1)
  if all(map(i,:) == [1 1 1]) && ...
      ( i == size(map,1) || all(map(i+1,:) == [0 0 0]) )
    idx = i;
    break;
  end
end

mask = x==(idx-1); % Zero based.
[r,c] = find(mask);

for i=1:length(r)
  iconRGB(r(i),c(i),:) = NaN;
end


