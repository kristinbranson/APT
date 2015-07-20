function sbfmf_write_nframes(fp,version,nframes)

fseek(fp,4+length(version)+4+4,'bof');
fwrite(fp,nframes,'uint32');