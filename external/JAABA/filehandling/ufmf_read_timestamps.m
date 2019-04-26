function [timestamps] = ufmf_read_timestamps(header,t0,t1)

FRAME_CHUNK = 1;

fp = header.fid;

if nargin < 2,
  t0 = 1;
end
if nargin < 3,
  t1 = header.nframes;
end

timestamps = nan(1,t1-t0+1);
for t = t0:t1,
  fseek(fp,header.frame2file(t),'bof');

  % read in the chunk type: 1
  chunktype = fread(fp,1,'uchar');
  if chunktype ~= FRAME_CHUNK,
    error('Expected chunktype = %d at start of frame, got %d',FRAME_CHUNK,chunktype);
  end
  % read in timestamp: 8
  timestamps(t-t0+1) = fread(fp,1,'double');
end
