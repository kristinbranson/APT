function [varargout] = read_mjpg_frame(headerinfo,f)

im = parsejpg8(headerinfo.filename,headerinfo.frame2file(f));
timestamp = headerinfo.timestamp(f);
varargout{1} = im;
if nargout >= 2,
  varargout{2} = timestamp;
end
