% crop_fmf(infmf,outfmf,startframe,endframe)
function crop_fmf(infmf,outfmf,startframe,endframe)

[header_size,version,nr,nc,bytes_per_chunk,nframes,data_format] = fmf_read_header(infmf);

if startframe < 1 || startframe > nframes || endframe < 0 || endframe > nframes || startframe > endframe,
    error('startframe or endframe is out of range.');
end

fin = fopen(infmf,'r');
%fout = fopen(outfmf,'w');
%buf = fread(fin,header_size,'uint8');
%fwrite(fout,buf);

fout = fmf_write_header(outfmf,header_size,version,nr,nc,bytes_per_chunk,endframe-startframe+1,data_format);

fseek(fin,header_size+(startframe-1)*bytes_per_chunk,'bof');
for f = startframe:endframe,
   if mod(f-startframe+1,100) == 0,
     % fprintf('Copying frame %d in range %d to %d\n',f,startframe,endframe);
   end
   buf = fread(fin,bytes_per_chunk,'uint8');
   fwrite(fout,buf);
end

fclose(fin);
fclose(fout);