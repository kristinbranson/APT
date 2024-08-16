function copyfileensuredir(src,dst,force)
% Throws if unsuccessful

if nargin < 3,
  force = true;
end

dstP = fileparts(dst);
if exist(dstP,'dir')==0
  [succ,msg] = mkdir(dstP);
  if ~succ
    error('Failed to create directory %s: %s',dstP,msg);
  end
end

if force || ~exist(dst,'file'),
  [succ,msg] = copyfile(src,dst);
  if ~succ
    error('Failed to copy %s to %s: %s',src,dst,msg);
  end
end