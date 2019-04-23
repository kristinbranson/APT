function [im,stamp] = fmfreadframe(fid,loc,nr,nc,bytes_per_chunk,data_format)

if fseek(fid,loc,'bof') == 0,
  [im,stamp] = fmf_read_frame(fid,nr,nc,bytes_per_chunk,data_format);
else
  im = -1;
  stamp = -1;
end