function v = imwriteCheck(im,filename,varargin)
imwrite(im,filename,varargin{:});
v = exist(filename,'file');
