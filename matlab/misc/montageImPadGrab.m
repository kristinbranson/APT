function imroi = montageImPadGrab(im,xc,yc,rad,th,tfAlign,padval)
  % grab a patch form an image centered around a certain point
  % im: image
  % xy/yc: center of patch
  % rad: radius
  % th: (optional) angle. only used if tfAlign
  % tfAlign: logical. if true, grab patch with rot as given by th
  % padval: background for padgrab

  if tfAlign
    % im: cropped + canonically rotated
    imnr = size(im,1);
    imnc = size(im,2);
    xim = 1:imnc;
    yim = 1:imnr;
    [xgim,ygim] = meshgrid(xim,yim);
    xroictr = -rad:rad;
    yroictr = -rad:rad;
    [xgroi,ygroi] = meshgrid(xroictr,yroictr);        
    imroi = readpdf2chan(DataAugMontage.convertIm2Double(im),...
      xgim,ygim,xgroi,ygroi,xc,yc,th);
  else
    % im: crop around current target, no rotation
    [roiXlo,roiXhi,roiYlo,roiYhi] = xyRad2roi(xc,yc,rad);
    imroi = padgrab2d(im,padval,roiYlo,roiYhi,roiXlo,roiXhi);
  end
end
