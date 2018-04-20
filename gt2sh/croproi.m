function [Ic,xyc] = croproi(I,xy,roi)
% I: [nx1] column image vec
% xy: [nxnptx2], 3rd dim is x/y
% roi: [nx4] [xlo xhi ylo yi] roi crop coords (inclusive)

assert(iscolumn(I));
n = numel(I);
npt = size(xy,2);
szassert(xy,[n npt 2]);
szassert(roi,[n 4]);

Ic = cell(size(I));
xyc = nan(size(xy));
for i=1:n
  xlo = roi(i,1);
  xhi = roi(i,2);
  ylo = roi(i,3);
  yhi = roi(i,4);
  Ic{i} = I{i}(ylo:yhi,xlo:xhi);
  xyc(i,:,1) = xy(i,:,1)-xlo+1;
  xyc(i,:,2) = xy(i,:,2)-ylo+1;
  if any(xyc(i,:,1)>xhi)
    warningNoTrace('Shape OOB in xdir: row %d\n',i);
  end
  if any(xyc(i,:,2)>yhi)
    warningNoTrace('Shape OOB in ydir: row %d\n',i);
  end
end  
