function sampling = getReduceSampling(image_dim,sample_factor)
% getReduceSampling gets the actual pixel indices we will sample when using
%   the 'Reduce' parameter in imshow.  image_dim is the size of the image
%   in a particular dimension (width or height).
% 
%   Examples
%   --------
%   sampled_rows = getReduceSampling(image_info.Height,sampleFactor);
%   sampled_cols = getReduceSampling(image_info.Width,sampleFactor);
%
%   See also IMSHOW.

%   Copyright 2007 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2007/12/10 21:38:15 $

% select start/end pixels such that the image pixel extent does not extend
% beyond the original pixel extent
start_pixel   = floor(sample_factor/2+1);
end_pixel     = ceil(image_dim-sample_factor/2);
initial_range = start_pixel:sample_factor:end_pixel;
end_pixel     = initial_range(end);

% find the total number of image pixels clipped while sampling
total_clipped_pixels = (start_pixel-1) + (image_dim - end_pixel);
start_buffer = floor(total_clipped_pixels / 2);

% translate the sampling to roughly center the total pixel span
sampling = initial_range - start_pixel + start_buffer + 1;
