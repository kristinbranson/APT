function imroi = padgrab2d(im,padval,r0,r1,c0,c1)
% stopgap generalization of padgrab
if ndims(im) == 2 %#ok<ISMAT>
  imroi = padgrab(im,padval,r0,r1,c0,c1);
elseif ndims(im) == 3
  imroi = padgrab(im,padval,r0,r1,c0,c1,1,size(im,3));
else
  error('Undefined number of channels');
end
