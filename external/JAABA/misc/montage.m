function [h,b] = montage(varargin)
%MONTAGE Display multiple image frames as rectangular montage.
%   MONTAGE displays all the frames of a multiframe image array in a single
%   image object, arranging the frames so that they roughly form a square.
%
%   MONTAGE(I) displays the K frames of the intensity image array I. I is
%   M-by-N-by-1-by-K.
%
%   MONTAGE(BW) displays the K frames of the binary image array BW. BW is
%   M-by-N-by-1-by-K.
%
%   MONTAGE(X,MAP) displays the K frames of the indexed image array X, using the
%   colormap MAP for all frames. X is M-by-N-by-1-by-K.
%
%   MONTAGE(RGB) displays the K frames of the truecolor image array RGB. RGB is
%   M-by-N-by-3-by-K.
%
%   H = MONTAGE(...) returns the handle to the image object.
%
%   Class support
%   -------------  
%   An intensity image can be logical, uint8, uint16, int16, single, or double.
%   An indexed image can be logical, uint8, uint16, single, or double. The map
%   must be double. A truecolor image can be uint8, uint16, single, or
%   double. The output is a handle to the graphics objects produced by
%   this function.
%
%   Example
%   -------
%       load mri
%       montage(D,map)
%
%   See also IMMOVIE.

%   Copyright 1993-2005 The MathWorks, Inc.
%   $Revision: 1.1.8.4 $  $Date: 2005/11/15 01:03:29 $

[a,cm,aspectRatio] = parse_inputs(varargin{:});

[nRows, nCols, nBands, nFrames] = size(a);

% Estimate nMontageColumns and nMontageRows given the desired ratio of
% Columns to Rows to be one (square montage).
nMontageCols = sqrt(aspectRatio * nRows * nFrames / nCols);

% Make sure montage rows and columns are integers. The order in the adjustment
% matters because the montage image is created horizontally across columns.
nMontageCols = ceil(nMontageCols); 
nMontageRows = ceil(nFrames / nMontageCols);

% Create the montage image.
b = a(1,1); % to inherit type 
b(1,1) = 0; % from a
b = repmat(b, [nMontageRows*nRows, nMontageCols*nCols, nBands, 1]);

rows = 1 : nRows; 
cols = 1 : nCols;

for i = 0:nMontageRows-1
  for j = 0:nMontageCols-1,
    k = j + i * nMontageCols + 1;
    if k <= nFrames
      b(rows + i * nRows, cols + j * nCols, :) = a(:,:,:,k);
    else
      break;
    end
  end
end

if isempty(cm)
  hh = imshow(b);
else
  hh = imshow(b,cm);
end

if nargout > 0
    h = hh;
end


%--------------------------------------------------------------
%Parse Inputs Function

function [I,map,aspectRatio] = parse_inputs(varargin)

% initialize variables
map = [];
aspectRatio = 1;

iptchecknargin(1,4,nargin,mfilename);
iptcheckinput(varargin{1},{'uint8' 'double' 'uint16' 'logical' 'single' ...
                    'int16'},{},mfilename, 'I, BW, or RGB',1);
I = varargin{1};

if nargin>=2
  if isa(I,'int16')
    eid = sprintf('Images:%s:invalidIndexedImage',mfilename);
    msg1 = 'An indexed image can be uint8, uint16, double, single, or ';
    msg2 = 'logical.';
    error(eid,'%s %s',msg1, msg2);
  end
  if nargin >= 3 & isstr(varargin{2}),
    if strcmp(varargin{2},'aspectratio'),
      aspectRatio = varargin{3};
    end;
  else,
    map = varargin{2};
    iptcheckinput(map,{'double'},{},mfilename,'MAP',1);
    if ((size(map,1) == 1) && (prod(map) == numel(I)))
      % MONTAGE(D,[M N P]) OBSOLETE
      eid = sprintf('Images:%s:obsoleteSyntax',mfilename);
      msg1 = 'MONTAGE(D,[M N P]) is an obsolete syntax.';
      msg2 = 'Use multidimensional arrays to represent multiframe images.';
      error(eid,'%s\n%s',msg1,msg2);    
    end
  end
end;
