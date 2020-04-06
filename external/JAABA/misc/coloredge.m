function [eout,thresh] = coloredge(a,varargin)
%COLOREDGE Find edges in intensity image.
%   EDGE takes a color image I as its input, and returns a 
%   binary image BW of the same size as I, with 1's where the function 
%   finds edges in I and 0's elsewhere.
%
%   COLOREDGE uses the Canny method for edge detection:
%
%      The Canny method finds edges by looking for local maxima of the
%      gradient of I. The gradient is calculated using the derivative of a
%      Gaussian filter. The method uses two thresholds, to detect strong
%      and weak edges, and includes the weak edges in the output only if
%      they are connected to strong edges. This method is therefore less
%      likely than the others to be "fooled" by noise, and more likely to
%      detect true weak edges.
%
%   Usage: 
%   BW = COLOREDGE(I), where I is the nrows x ncols x ncolors image
%
%   BW = COLOREDGE(I,PARAM1,VAL1,PARAM2,VAL2,...) specifies parameters of
%   the edge detection:
%   'thresh'        specifies sensitivity thresholds for the Canny method.
%                   THRESH is a two-element vector in which the first 
%                   element is the low threshold, and the second element 
%                   is the high threshold. If you specify a scalar for 
%                   THRESH, this value is used for the high threshold and 
%                   0.4*THRESH is used for the low threshold. If you do not
%                   specify THRESH, or if THRESH is empty ([]), EDGE 
%                   chooses low and high  values automatically.
%
%   'sigma'         specifies using SIGMA as the standard deviation of the 
%                   Gaussian filter. The default SIGMA is 2; the size of 
%                   the filter is chosen automatically, based on SIGMA. 
%
%   [BW,thresh] = COLOREDGE(I,...) returns the threshold values as a
%   two-element vector.
%
%   Class Support
%   -------------
%   I is a nonsparse numeric array. BW is of class logical.
%
%   Example
%   -------
%   Find the edges of the circuit.tif image using the Prewitt and Canny
%   methods:
%
%       I = imread('board.tif');
%       BW = coloredge(I);
%       figure, imshow(BW)
%
%   See also FSPECIAL.

thresh = []; sigma = 2;
for i = 1:2:length(varargin),
  switch varargin{i},
    case 'thresh',
      thresh = varargin{i+1};
    case 'sigma',
      sigma = varargin{i+1};
  end;
end;

% Transform to a double precision intensity image if necessary
if ~isa(a,'double') && ~isa(a,'single') 
  a = im2single(a);
end

[m,n,ncolors] = size(a);

% The output edge map:
e = false(m,n);

% Magic numbers
GaussianDieOff = .0001;
PercentOfPixelsNotEdges = .7; % Used for selecting thresholds
ThresholdRatio = .4;          % Low thresh is this fraction of the high.

% Design the filters - a gaussian and its derivative

pw = 1:30; % possible widths
ssq = sigma^2;
width = find(exp(-(pw.*pw)/(2*ssq))>GaussianDieOff,1,'last');
if isempty(width)
  width = 1;  % the user entered a really small sigma
end

t = (-width:width);
gau = exp(-(t.*t)/(2*ssq))/(2*pi*ssq);     % the gaussian 1D filter

% Find the directional derivative of 2D Gaussian (along X-axis)
% Since the result is symmetric along X, we can get the derivative along
% Y-axis simply by transposing the result for X direction.
[x,y]=meshgrid(-width:width,-width:width);
dgau2D=-x.*exp(-(x.*x+y.*y)/(2*ssq))/(pi*ssq);

% Convolve the filters with the image in each direction
% The canny edge detector first requires convolution with
% 2D gaussian, and then with the derivitave of a gaussian.
% Since gaussian filter is separable, for smoothing, we can use
% two 1D convolutions in order to achieve the effect of convolving
% with 2D Gaussian.  We convolve along rows and then columns.

%smooth the image out
aSmooth=imfilter(a,gau,'conv','replicate');   % run the filter accross rows
aSmooth=imfilter(aSmooth,gau','conv','replicate'); % and then accross columns

%apply directional derivatives
ax = imfilter(aSmooth, dgau2D, 'conv','replicate');
ay = imfilter(aSmooth, dgau2D', 'conv','replicate');

%J(k,1,i,j)=ax(i,j,k), J(k,2,i,j)=ay(i,j,k)
%J'*J = [\sum_k J(k,1)^2           \sum_k J(k,1)*J(k,2)]
%       [\sum_k J(k,1)*J(k,2)      \sum_k J(k,2)^2     ]
%     = [sum(ax.^2,3)              sum(ax.*ay,3)]
%       [sum(ax.*ay,3)             sum(ay.^2,3) ]
a1 = sum(ax.^2,3);
a2 = sum(ax.*ay,3);
a4 = sum(ay.^2,3);
mag = ( (a1+a4) + sqrt( (a1+a4).^2 + 4*(a2.^2 - a1.*a4) ) ) / 2;
vx = a2;
vy = mag - a1;
z = sqrt(vx.^2 + vy.^2);
ax = vx ./ z;
ay = vy ./ z;

magmax = max(mag(:));
if magmax>0
  mag = mag / magmax;   % normalize
end

% Select the thresholds
if isempty(thresh)
  counts=imhist(mag, 64);
  highThresh = find(cumsum(counts) > PercentOfPixelsNotEdges*m*n,...
    1,'first') / 64;
  lowThresh = ThresholdRatio*highThresh;
  thresh = [lowThresh highThresh];
elseif length(thresh)==1
  highThresh = thresh;
  if thresh>=1
    eid = sprintf('Images:%s:thresholdMustBeLessThanOne', mfilename);
    msg = 'The threshold must be less than 1.';
    error(eid,'%s',msg);
  end
  lowThresh = ThresholdRatio*thresh;
  thresh = [lowThresh highThresh];
elseif length(thresh)==2
  lowThresh = thresh(1);
  highThresh = thresh(2);
  if (lowThresh >= highThresh) || (highThresh >= 1)
    eid = sprintf('Images:%s:thresholdOutOfRange', mfilename);
    msg = 'Thresh must be [low high], where low < high < 1.';
    error(eid,'%s',msg);
  end
end

% The next step is to do the non-maximum supression.
% We will accrue indices which specify ON pixels in strong edgemap
% The array e will become the weak edge map.
idxStrong = [];
for dir = 1:4
  idxLocalMax = cannyFindLocalMaxima(dir,ax,ay,mag);
  idxWeak = idxLocalMax(mag(idxLocalMax) > lowThresh);
  e(idxWeak)=1;
  idxStrong = [idxStrong; idxWeak(mag(idxWeak) > highThresh)];
end

if ~isempty(idxStrong) % result is all zeros if idxStrong is empty
  rstrong = rem(idxStrong-1, m)+1;
  cstrong = floor((idxStrong-1)/m)+1;
  e = bwselect(e, cstrong, rstrong, 8);
  e = bwmorph(e, 'thin', 1);  % Thin double (or triple) pixel wide contours
end

if nargout==0,
  imshow(e);
else
  eout = e;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function : cannyFindLocalMaxima
%
function idxLocalMax = cannyFindLocalMaxima(direction,ix,iy,mag)
%
% This sub-function helps with the non-maximum supression in the Canny
% edge detector.  The input parameters are:
% 
%   direction - the index of which direction the gradient is pointing, 
%               read from the diagram below. direction is 1, 2, 3, or 4.
%   ix        - input image filtered by derivative of gaussian along x 
%   iy        - input image filtered by derivative of gaussian along y
%   mag       - the gradient magnitude image
%
%    there are 4 cases:
%
%                         The X marks the pixel in question, and each
%         3     2         of the quadrants for the gradient vector
%       O----0----0       fall into two cases, divided by the 45 
%     4 |         | 1     degree line.  In one case the gradient
%       |         |       vector is more horizontal, and in the other
%       O    X    O       it is more vertical.  There are eight 
%       |         |       divisions, but for the non-maximum supression  
%    (1)|         |(4)    we are only worried about 4 of them since we 
%       O----O----O       use symmetric points about the center pixel.
%        (2)   (3)        


[m,n] = size(mag);

% Find the indices of all points whose gradient (specified by the 
% vector (ix,iy)) is going in the direction we're looking at.  

switch direction
 case 1
  idx = find((iy<=0 & ix>-iy)  | (iy>=0 & ix<-iy));
 case 2
  idx = find((ix>0 & -iy>=ix)  | (ix<0 & -iy<=ix));
 case 3
  idx = find((ix<=0 & ix>iy) | (ix>=0 & ix<iy));
 case 4
  idx = find((iy<0 & ix<=iy) | (iy>0 & ix>=iy));
end

% Exclude the exterior pixels
if ~isempty(idx)
  v = mod(idx,m);
  idx(v==1 | v==0 | idx<=m | (idx>(n-1)*m)) = [];
end

ixv = ix(idx);  
iyv = iy(idx);   
gradmag = mag(idx);

% Do the linear interpolations for the interior pixels
switch direction
 case 1
  d = abs(iyv./ixv);
  gradmag1 = mag(idx+m).*(1-d) + mag(idx+m-1).*d; 
  gradmag2 = mag(idx-m).*(1-d) + mag(idx-m+1).*d; 
 case 2
  d = abs(ixv./iyv);
  gradmag1 = mag(idx-1).*(1-d) + mag(idx+m-1).*d; 
  gradmag2 = mag(idx+1).*(1-d) + mag(idx-m+1).*d; 
 case 3
  d = abs(ixv./iyv);
  gradmag1 = mag(idx-1).*(1-d) + mag(idx-m-1).*d; 
  gradmag2 = mag(idx+1).*(1-d) + mag(idx+m+1).*d; 
 case 4
  d = abs(iyv./ixv);
  gradmag1 = mag(idx-m).*(1-d) + mag(idx-m-1).*d; 
  gradmag2 = mag(idx+m).*(1-d) + mag(idx+m+1).*d; 
end
idxLocalMax = idx(gradmag>=gradmag1 & gradmag>=gradmag2); 