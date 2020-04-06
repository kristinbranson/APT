function [hhead,hline] = vectarrow(p0,p1,arrowheadlength,arrowheadwidth,varargin)
%Arrowline 3-D vector plot.
%   vectarrow(p0,p1) plots a line vector with arrow pointing from point p0
%   to point p1. The function can plot both 2D and 3D vector with arrow
%   depending on the dimension of the input
%
%   Example:
%       3D vector
%       p0 = [1 2 3];   % Coordinate of the first point p0
%       p1 = [4 5 6];   % Coordinate of the second point p1
%       vectarrow(p0,p1)
%
%       2D vector
%       p0 = [1 2];     % Coordinate of the first point p0
%       p1 = [4 5];     % Coordinate of the second point p1
%       vectarrow(p0,p1)
%
%   See also Vectline

%   Rentian Xiong 4-18-05
%   $Revision: 1.0

if ~exist('arrowheadlength','var') || isempty(arrowheadlength) || arrowheadlength == 0,
  alpha = .1;
else
  alpha = arrowheadlength;  % Size of arrow head relative to the length of the vector
end
if ~exist('arrowheadwidth','var') || isempty(arrowheadwidth) || arrowheadwidth == 0,
  beta = 0.1;  % Width of the base of the arrow head relative to the length
else
  beta = arrowheadwidth;
end

  if max(size(p0))==3
      if max(size(p1))==3
          x0 = p0(1);
          y0 = p0(2);
          z0 = p0(3);
          x1 = p1(1);
          y1 = p1(2);
          z1 = p1(3);
          hline = plot3([x0;x1],[y0;y1],[z0;z1],varargin{:});   % Draw a line between p0 and p1
          
          p = p1-p0;
          
          hu = [x1-alpha*(p(1)+beta*(p(2)+eps)); x1; x1-alpha*(p(1)-beta*(p(2)+eps))];
          hv = [y1-alpha*(p(2)-beta*(p(1)+eps)); y1; y1-alpha*(p(2)+beta*(p(1)+eps))];
          hw = [z1-alpha*p(3);z1;z1-alpha*p(3)];
          
          hhead = plot3(hu(:),hv(:),hw(:),varargin{:})  % Plot arrow head
      else
          error('p0 and p1 must have the same dimension')
      end
  elseif max(size(p0))==2
      if max(size(p1))==2
          x0 = p0(1);
          y0 = p0(2);
          x1 = p1(1);
          y1 = p1(2);
          hline = plot([x0;x1],[y0;y1],varargin{:});   % Draw a line between p0 and p1
          
          p = p1-p0;
          
          hu = [x1-alpha*(p(1)+beta*(p(2)+eps)); x1; x1-alpha*(p(1)-beta*(p(2)+eps))];
          hv = [y1-alpha*(p(2)-beta*(p(1)+eps)); y1; y1-alpha*(p(2)+beta*(p(1)+eps))];
          
          hold on
          hhead = plot(hu(:),hv(:),varargin{:})  % Plot arrow head
      else
          error('p0 and p1 must have the same dimension')
      end
  else
      error('this function only accepts 2D or 3D vector')
  end