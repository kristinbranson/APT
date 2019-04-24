function nboxes = ufmf_read_nboxes(header,frameis)

FRAME_CHUNK = 1;

fp = header.fid;

nboxes = nan(size(frameis));
for i = 1:numel(frameis),
  
  framei = frameis(i);

  fseek(fp,header.frame2file(framei),'bof');

  % read in the chunk type: 1
  chunktype = fread(fp,1,'uchar');
  if chunktype ~= FRAME_CHUNK,
    error('Expected chunktype = %d at start of frame, got %d',FRAME_CHUNK,chunktype);
  end
  % read in timestamp: 8
  fread(fp,1,'double');
  if header.version == 4,
    % number of points: 2
    npts = fread(fp,1,'uint32');
  else
    % number of points: 2
    npts = fread(fp,1,'ushort');
  end
  nboxes(i) = npts;
end