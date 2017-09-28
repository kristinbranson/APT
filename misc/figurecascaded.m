function h = figurecascaded(h0,varargin)
% Create new figure cascaded/offset from figure h0

pos = h0.Position;
pos([1 2]) = pos([1 2]) + pos([3 4]).*[.1 -.1];
h = figure('Position',pos,varargin{:});