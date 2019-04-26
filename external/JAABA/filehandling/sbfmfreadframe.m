% [im,stamp] = sbfmfreadframe(f,fid,frame2file,bgcenter)
function [im,stamp] = sbfmfreadframe(f,fid,frame2file,bgcenter)

if fseek(fid,frame2file(f),'bof') == 0,
  [im,stamp] = sbfmf_read_frame(fid,bgcenter);
else
  im = -1;
  stamp = -1;
end