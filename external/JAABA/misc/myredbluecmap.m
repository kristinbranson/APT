function p = myredbluecmap(m,varargin)
%MYREDBLUECMAP creates a red and green colormap.
%
%   MYREDBLUECMAP(M) returns an M-by-3 matrix containing a red and green
%   colormap. Low values are bright green, values in the center of the map
%   are black, and high values are red. If M is empty, the length of the
%   map will be the same as the length of the colormap of the current figure.
%
%   MYREDBLUECMAP(...,'INTERPOLATION',METHOD) allows you to set how the
%   colors are interpolated. Valid options are 'linear', 'quadratic',
%   'cubic', and 'sigmoid'. Default is 'linear'.
%
%   MYREDBLUECMAP, by itself, is the same length as the current colormap.
%
%   For example, to reset the colormap of the current figure, type
%
%             colormap(myredbluecmap)
%
%             % Use linear interpolation from red to black to green
%             colormap(myredbluecmap([],'interpolation','linear'))
%
%   See also CLUSTERGRAM, COLORMAP, COLORMAPEDITOR.

%   Copyright 2003-2006 The MathWorks, Inc.

maxncolors_redblue = 11;

if nargin < 1 || isempty(m)
    m = size(get(gcf,'colormap'),1);
end

if m <= maxncolors_redblue,
  p = redbluecmap(m,varargin{:});
  return;
end

p0 = redbluecmap(maxncolors_redblue);

p = nan(m,3);
for i = 1:3,
  p(:,i) = interp1(linspace(0,1,maxncolors_redblue)',p0(:,i),linspace(0,1,m)');
end
% 
% coloredLength = floor((m-1)/2);
% 
% % deal with small inputs in a consistent way
% if m < 3
%     if m == 0
%         p = zeros(0,3);
%     elseif m == 1
%         p = zeros(1,3);
%     else
%         p = [0 0 1; 1 0 0];
%     end
%     return
% end
% 
% interpMethod = 'linear';
% % get input arguments
% if  nargin > 1
%     if rem(nargin,2) ~= 1
%         error(message('fastaread:IncorrectNumberOfArguments', mfilename));
%     end
%     okargs = {'interpolation',''};
%     for j=1:2:nargin-1
%         pname = varargin{j};
%         pval = varargin{j+1};
%         k = find(strncmpi(pname,okargs,numel(pname)));
%         if isempty(k)
%             error(message('fastaread:UnknownParameterName', pname));
%         else
%             switch(k)
%                 case 1  % ignore gaps
%                     interpMethod = pval;
%             end
%         end
%     end
% end
% 
% % create an appropriately long linearly interpolated chunk
% interpCol = (1/(coloredLength):1/(coloredLength):1);
% 
% okmethods = {'sigmoid','linear','quadratic','cubic'};
% theMethod = find(strncmpi(interpMethod,okmethods,numel(interpMethod)));
% if numel(theMethod) ~= 1
%     error(message('myredbluecmap:InvalidMethod', interpMethod));
% else
%     switch(theMethod)
%         case 1 %tanh
%             interpCol = tanh(pi*(interpCol));
%         case 2 % linear
%             % Don't need to do anything
%         case 3 %quadratic
%             interpCol = (interpCol).^(1/2);
%         case 4 %cubic
%             interpCol = (interpCol).^(1/3);
%     end
% end
% 
% if coloredLength == ((m-1)/2)
%     fillerZeros = 0;
% else
%     fillerZeros = [0 0];
% end
% 
% % red is linear for red half
% red = fliplr(1-[zeros(size(interpCol)) fillerZeros interpCol]);
% % blue is opposite of red
% blue = fliplr(red);
% % green is lowintensity for red half
% green = min(red,blue);
% 
% p = [red',green',blue'];
