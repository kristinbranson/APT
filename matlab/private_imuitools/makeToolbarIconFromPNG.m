function icon = makeToolbarIconFromPNG(filename)
% makeToolbarIconFromPNG  Creates an icon with transparent
%   background from a PNG image.

%   Copyright 2004 The MathWorks, Inc.  
%   $Revision: 1.1.8.1 $  $Date: 2004/08/10 01:50:31 $

  % Read image and alpha channel if there is one.
  [icon,map,alpha] = imread(filename);

  % If there's an alpha channel, the transparent values are 0.  For an RGB
  % image the transparent pixels are [0, 0, 0].  Otherwise the background is
  % cyan for indexed images.
  if (ndims(icon) == 3) % RGB

    idx = 0;
    if ~isempty(alpha)
      mask = alpha == idx;
    else
      mask = icon==idx; 
    end
    
  else % indexed
    
    % Look through the colormap for the background color.
    for i=1:size(map,1)
      if all(map(i,:) == [0 1 1])
        idx = i;
        break;
      end
    end
    
    mask = icon==(idx-1); % Zero based.
    icon = ind2rgb(icon,map);
    
  end
  
  % Apply the mask.
  icon = im2double(icon);
  
  for p = 1:3
    
    tmp = icon(:,:,p);
    tmp(mask) = NaN;
    icon(:,:,p) = tmp;
    
  end
