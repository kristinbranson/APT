% [im,stamp] = sbfmf_read_frame(fp,bgcenter)
function [im,stamp] = sbfmf_read_frame(fp,bgcenter)

npixels = double(fread(fp,1,'uint32'));
stamp = fread(fp,1,'double');
idx = double(fread(fp,npixels,'uint32'))+1;
val = double(fread(fp,npixels,'uint8'));
im = bgcenter;
im(idx) = val;
im = im';