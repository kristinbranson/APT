function [isXDataDefault,isYDataDefault] = isDefaultXYData(himage)
%isDefaultXYData Checks if image object has default XData and YData.
%   isDefaultXYData(himage) checks that an image object, HIMAGE, has default
%   XData and YData.  HIMAGE must be a valid image handle object.

%   Copyright 1993-2004 The MathWorks, Inc.  
%   $Revision: 1.1.8.1 $  $Date: 2004/08/10 01:50:24 $

[imageHeight,imageWidth,p] = size(get(himage,'CData'));
defaultXData = [1 imageWidth];
defaultYData = [1 imageHeight];   

isXDataDefault = isequal(get(himage,'XData'),defaultXData);
isYDataDefault = isequal(get(himage,'YData'),defaultYData);

