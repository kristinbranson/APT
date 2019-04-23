function [hEllipse,X,Y] = ellipsearcdraw(theta0,theta1,a,b,x0,y0,phi,lineStyle)
%ELLIPSEDRAW can draw an arbitrary ellipse with given parameters.
%   The properties of that ellipse plot can be customized 
%   by setting the ellipse handle. 
%
%       hEllipse = ellipsearcdraw(theta0,theta1,a,b,x0,y0,phi,lineStyle)
%
%   Input parameters:
%       theta0      Start of arc
%       theta1      End of arc
%       a           Value of the major axis
%       b           Value of the minor axis
%       x0          Abscissa of the center point of the ellipse
%       y0          Ordinate of the center point of the ellipse
%       phi         Angle between x-axis and the major axis
%       lineStyle   Definition of the plotted line style
%
%   Output:
%       hEllipse    Handle of the ellipse
%
%   Simple usage:
%       ellipsedraw(5,3);
%       ellipsedraw(5,3,'g--');
%       ellipsedraw(5,3,pi/4);
%
%   Complete usage:
%       h = ellipsedraw(5,3,1,-2,pi/4,'r-.');
%       set(h,'LineWidth',2);

% Designed by: Lei Wang, <WangLeiBox@hotmail.com>, 25-Mar-2003.
% Last Revision: 01-Apr-2003.
% Dept. Mechanical & Aerospace Engineering, NC State University.
% Copyright (c)2003, Lei Wang <WangLeiBox@hotmail.com>
%$Revision: 1.1 $  $ 4/1/2003 5:42:24 PM $

if (nargin < 4)|(nargin > 8),
    error('Please see help for INPUT DATA.');
    
elseif nargin == 4
    x0 = 0;     y0 = 0;
    phi = 0;    lineStyle = 'b-';
    
elseif nargin == 5
    if ischar(x0) == 1
        lineStyle = x0;         
        x0 = 0; y0 = 0;
        phi = 0; 
    else
        phi = x0;  
        x0 = 0; y0 = 0;
        lineStyle = 'b-';
    end
    
elseif nargin == 6    
    phi = 0;    lineStyle = 'b-';
    
elseif nargin == 7
    lineStyle = 'b-';
end


theta = linspace(theta0,theta1,100);

% Parametric equation of the ellipse
%----------------------------------------
 x = a*cos(theta);
 y = b*sin(theta);



% Coordinate transform 
%----------------------------------------
 X = cos(phi)*x - sin(phi)*y;
 Y = sin(phi)*x + cos(phi)*y;
 X = X + x0;
 Y = Y + y0;


% Plot the ellipse
%----------------------------------------
 hEllipse = plot(X,Y,lineStyle);