function [s,sz,unit] = getFileSizeStr(bytes,fmt)

if nargin < 2,
  fmt = '%.1f %s';
end

unex = log2(bytes);
if unex < 10,
  unit = 'B';
  sz = bytes;
elseif unex < 20,
  unit = 'KB';
  sz = bytes/2^10;
elseif unex < 30,
  unit = 'MB';
  sz = bytes/2^20;
elseif unex < 40,
  unit = 'GB';
  sz = bytes/2^30;
else
  unit = 'TB';
  sz = bytes/2^40;
end

s = sprintf(fmt,sz,unit);