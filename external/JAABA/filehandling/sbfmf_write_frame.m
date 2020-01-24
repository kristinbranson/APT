function frameloc = sbfmf_write_frame(fp,stamp,idx,val)

frameloc = ftell(fp);
npixels = length(idx);
fwrite(fp,npixels,'uint32');
fwrite(fp,stamp,'double');
fwrite(fp,uint32(idx-1),'uint32');
fwrite(fp,uint8(val),'uint8');
