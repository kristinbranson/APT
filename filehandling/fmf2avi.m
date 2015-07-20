function fmf2avi(infilename,outfilename,nframes,scale)

aviobj = avifile(outfilename);
[header_size, version, f_height, f_width, bytes_per_chunk, ...              
 max_n_frames, data_format] = fmf_read_header(infilename);
fp = fopen(infilename,'r');
fseek(fp,header_size,'bof');
for i = 1:nframes,
  fprintf('%d\n',i);
  data = fmf_read_frame(fp,f_height,f_width,bytes_per_chunk, ...
                        data_format);
  data = imresize(uint8(data),scale);
  aviobj = addframe(aviobj,repmat(data,[1,1,3]));
end;

aviobj = close(aviobj);