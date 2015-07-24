  function min_zoom =  getMinimumPixelRegionMag(viewport,hIm)
  %getMinimumPixelRegionMag returns the minimum magnification enforced in
  %impixelregion and impixelregionpanel MIN_ZOOM = getMinimumPixelRegionMag
  % returns the minimum magnification such that the pixel region rectangle
  % will fit inside the image boundaries.
      
[im_width,im_height] = size(get(hIm,'CData'));
min_zoom = max(viewport(1)/im_width,viewport(2)/im_height);
      

