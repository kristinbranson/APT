function deltaOnePixel = getDeltaOnePixel(dimData,imDim)
% Calculate the extent of one pixel in terms of the user units as defined by
% dimData which will be either the 'XData' or 'YData' associated with an image.

delta = dimData(2) - dimData(1);
if (imDim ~= 1)
    deltaOnePixel = delta/(imDim-1);
else
    deltaOnePixel = 1;
end