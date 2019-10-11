function [x, y] = ellipseToPolygon(a, b, xc, yc, delta)
%ellipseToPolygon Approximate ellipse as polygon
%
%   [x, y] = ellipseToPolygon(a, b, xc, yc, delta) constructs a polygon
%   that approximates an ellipse with semi-axes lengths a and b,
%   centered at (xc, yc).  Vectors x and y contain the coordinates of
%   the polygon vertices.  The ellipse is assumed to align with the
%   coordinate axes such that the length of the axis parallel to the
%   x-axis is 2a and the length of the axis parallel to the y-axis is
%   2b.  That is, the ellipse satisfies the equation:
%
%            ((x - xc)/a)^2 + ((y - yc)/b)^2 = 1.
%
%   The parameter delta specifies how closely the polygon (x, y) matches
%   the ellipse.  Every vertex in the polygon falls precisely on the
%   ellipse itself.  Between each pair of vertices is a polygon edge
%   that defines a chord within the ellipse.  The number and placement
%   of the vertices are chosen automatically such that (1) the maximum
%   departure of a given chord from the ellipse itself never exceeds
%   delta and (2) the size of the departure is roughly the same for
%   all edges of the polygon.
%
%   An additional symmetry constraint is imposed as well:  the polygon
%   vertices will always include the four points at the ends of the
%   axes:  (xc + a, yc), (xc, yc + b), (xc - a, yc), (xc, yc - b)).
%
%   For sufficiently small values of delta, the actual departures
%   closely approximate delta itself.  However, if delta is large with
%   respect to the axes lengths, the symmetry constraint results in a
%   minimum of four vertices and a natural upper bound on the departure.
%
%   See also ELLIPSE1 in Mapping Toolbox.

% Copyright 2007 The MathWorks, Inc.
% $Revision: 1.1.6.2 $  $Date: 2007/06/04 21:11:13 $

% This function uses some of the same principles and techniques as
% ELLIPSE1 in Mapping Toolbox, but with significant differences also.

% Compute the parametric sampling density at an closely-spaced sequence
% of points in the first quadrant.  The points are evenly spaced in the
% parameter t, where t is a parameter such that:
%
%              (xc + a * cos(t), yc + b *sin(t))
%
% is a point on the ellipse.  nu(k) is the parametric sampling density
% at t(k) chosen according to the criteria above.
M = 128;  % Choose enough points to ensure a smooth sampling
[nu, t] = parametricSamplingDensity(a, b, delta, M);

% Integrate nu using the trapezoid rule, including partial integrals.
% S(k) will approximate the integral of nu(t) from t=t1 to t=t(k), over
% (k-1) trapezoids each having width t(2) - t(1).
S = (t(2) - t(1)) * (2*cumsum(nu) - nu - nu(1)) / 2;

% Get N, the approximate number of points needed to cover t = 0 to
% t = pi/2 with a maximum departure approximating delta, then
% renormalize the partial integrals such that S(end) == N.
N = 1 + ceil(S(end));
S = N * (S / S(end));

% Interpolate to obtain sampling points:  Find the values of t at which
% S is an integer in the interval [0 (N-1)].  The result should fall one
% sample short of pi/2.  Preprocess by removing repeating values from S
% (in case the ellipse is flat: a == 0 or b == 0), then interpolate.
[S,indx] = unique(S);
t = t(indx);
ts = interp1q(S(:),t(:),(1:(N-1))')';

% Expand ts to cover all four quadrants.
tsflip = ts((N-1):-1:1);
ts = [0 ts pi/2 (pi - tsflip) pi (pi + ts) 3*pi/2 (2*pi - tsflip) 0];

% Compute the sampling points, accounting for the center offset.
x = xc + a * cos(ts);
y = yc + b * sin(ts);

%-----------------------------------------------------------------------

function [nu, t] = parametricSamplingDensity(a, b, delta, M)

% Compute the parametric sampling density nu at M+1 points corresponding
% regularly-spaced values of the parameter t ranging from 0 to pi/2.
% Approximate the sampling density that evenly distributes, across all
% the chords connecting adjacent sampling points, the maximum departure
% from the ellipse of the chord from the ellipse itself, with the
% maximum departure approximating delta.

% Divide the interval [0 pi/2] into M intervals of equal width.
dt = (pi/2) / M;

% Compute the value of the parameter t at each of the (M-1) intermediate
% points:
t = (1:(M-1)) * dt;

% The basic formula for nu,
%
%     sqrt((a*b)/(8*delta)) * ((a * sin(t)).^2 + (b * cos(t)).^2).^(-1/4)
%
% cannot be used if either a or b is zero, so use an alternate
% formulation that is robust at intermediate points:
nu = sqrt(1/(8*delta)) * ((sin(t)/b).^2 + (cos(t)/a).^2).^(-1/4);

% Then apply formulas specific to t == 0 and t == pi/2 and concatenate
% the results.
t =  [        0            t          pi/2        ];
nu = [sqrt((a)/(8*delta))  nu  sqrt((b)/(8*delta))];
