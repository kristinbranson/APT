function ims = readAllFrames(mov,n)
% read all frames in mov from 1:n
%
% ims: [nx1] cell array of ims

vr = VideoReader(mov);
ims = cell(n,1);
for i=1:n
  im = vr.readFrame();
  if size(im,3)>1
    assert(size(im,3)==3);
    assert(isequal(im(:,:,1),im(:,:,2),im(:,:,3)));
    im = im(:,:,1);
  end
  ims{i} = im;
end
