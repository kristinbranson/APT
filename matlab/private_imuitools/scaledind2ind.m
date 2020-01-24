function [X_new,map] = scaledind2ind(X,map,clim)
%SCALEDIND2IND Convert scaled indexed image to indexed image.
%   X_NEW = SCALEDIND2IND(X,MAP,CLIM) converts a scaled indexed image X
%   with associated colormap MAP and axes clim CLIM to an indexed image
%   X_NEW. The values in X_NEW are direct indices into the colormap.
% 
%   Class Support
%   -------------      
%   The input image I can be logical, uint8, uint16, int16, single, or
%   double and must be real and nonsparse.  I can have any dimension.  The
%   class of the output image X_NEW is uint8 if the colormap length is less
%   than or equal to 256; otherwise it is uint16.
%
%   Example
%   -------
%       h_im = imshow('rice.png');
%       colormap jet
%       X = get(h_im,'CData');
%       map = get(gcf,'Colormap');
%       clim = get(gca,'CLim');
%       [X,map] = scaledind2ind(X,map,clim);
%       figure, imshow(X, map);
%
%   See also GRAY2IND, IND2RGB.

%   Copyright 2008 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2008/02/07 16:30:46 $


    num_rows_cmap = size(map,1);
    
    % For integer indexed images, 0 corresponds to the first entry in the
    % colormap. Interpolate to find Y values which are the direct indices
    % into the given colormap.
    
    %yi = interp1(x,Y,xi,method,'extrap')
    X_new = interp1(clim,[0 num_rows_cmap-1],double(X(:)),'linear','extrap');
    
    X_new( X_new <0) = 0;
    X_new( X_new > num_rows_cmap-1) = num_rows_cmap-1;
    
    X_new = reshape(X_new,size(X));
    
    if num_rows_cmap <= 256
        X_new = uint8(X_new);
    else
        X_new = uint16(X_new);
    end
    


