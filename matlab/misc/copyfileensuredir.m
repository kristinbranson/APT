function copyfileensuredir(src,dst)
% Throws if unsuccessful

dstP = fileparts(dst);
if exist(dstP,'dir')==0
  [succ,msg] = mkdir(dstP);
  if ~succ
    error('Failed to create directory %s: %s',dstP,msg);
  end
end

[succ,msg] = copyfile(src,dst);
if ~succ
  error('Failed to copy %s to %s: %s',src,dst,msg);
end