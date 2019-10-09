function imageWidth = getImWidth(hIm)
%getImWidth returns the overall spatial width of an R-Set or non-R-Set
%image.
%
% imageWidth = getImWidth(hIm) returns the spatial width of the image
% hIm. 

%   Copyright 2008 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2008/11/24 14:58:37 $

if isRSetImage(hIm)
    imageWidth = getSpatialDims(hIm);
else
    img = get(hIm,'cdata');
    imageWidth = size(img,2);
end
