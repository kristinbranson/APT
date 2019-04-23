function imscwrite(I,name,varargin)

I = double(I);
c = colormap;
if ndims(I) ~= 2,
  error('input image must be N x M');
end;

a = min(I(:));
b = max(I(:));
d = b-a;
n = rows(c);
J = round((I - a)/d*(n-1)+1);
Irgb = reshape(c(J(:),:),[rows(I),cols(I),3]);
imwrite(im2uint8(Irgb),name,varargin{:});