function A = normalize(A,maxV,method)
% Normalize array A for writing into frame
%
% USAGE
%  A = normalize(A,maxV,method)
%
% INPUTS
%  A             -  MxNxT Array of frames
%  maxV          -  [1] maximum value (1 or 255 typically)
%  method        -  [1] 
%                    1 normalizes in [0,maxV]
%                    2 applies square power first
%                    0 apply absolute value first
%
% OUTPUTS
%  A             -  MxNxT Array of normalized frames
%
% EXAMPLE
%  
% A = normalize(A,255,1)
%
% See also
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia

if(nargin<2), maxV = 1; method=1; end 
if(nargin<3), method = 1; end
if(method==0), A = abs(A);
elseif(method==2), A = A.^2; end
mn = min(A(:));A = A - mn;
if(max(A(:))>0), A = (A./(max(A(:)))).*maxV; end
end