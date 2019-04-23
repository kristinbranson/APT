function frameloc = ufmf_write_frame(fp,stamp,x0,y0,w,h,val)

frameloc = ftell(fp);
% write the chunk id (points=1)
fwrite(fp,1,'uchar');
% write timestamp
fwrite(fp,stamp,'double');
% write number of points
npts = numel(x0);
fwrite(fp,npts,'uint32');

for i = 1:npts,
  % write x, y, width, height
  fwrite(fp,[x0(i),y0(i),w(i),h(i)],'ushort');
  % write region intensities
  fwrite(fp,val{i},'uint8');
end

