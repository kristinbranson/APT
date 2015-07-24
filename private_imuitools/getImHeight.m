function imageHeight = getImHeight(hIm)
%getImHeight returns the overall spatial height of an R-Set or non-R-Set
%image.
%
% imageWidth = getImWidth(hIm) returns the spatial height of the image
% hIm.

%   Copyright 2008 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2008/11/24 14:58:36 $

if isRSetImage(hIm)
    [imageWidth,imageHeight] = getSpatialDims(hIm);
else
    img = get(hIm,'cdata');
    imageHeight = size(img,1);
end