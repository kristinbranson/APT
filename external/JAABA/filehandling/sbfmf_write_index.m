function sbfmf_write_index(fp,framesloc,indexlocptr)

indexloc = ftell(fp);
fwrite(fp,uint64(framesloc),'uint64');
fseek(fp,indexlocptr,'bof');
fwrite(fp,uint64(indexloc),'uint64');