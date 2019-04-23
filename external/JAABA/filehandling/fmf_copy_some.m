function fmf_copy_some(infilename,outfilename,nframes,startframe,nskip)

if nargin < 4,
  startframe = 1;
end;
if nargin < 5,
  nskip = 1;
end;

% read in the header
[header_size, version, f_height, f_width, bytes_per_chunk, ...         
 max_n_frames, data_format] = fmf_read_header( infilename );
datatype = fmf_get_datatype(data_format);

% open the input file
fpin = fopen(infilename,'r');

% open the output file
if exist(outfilename,'file'),
  v = input(sprintf('%s exists. Overwrite? (y/n)',outfilename),'s');
  if lower(v) == 'y',
    fpout = fopen(outfilename,'w');
  else,
    return;
  end;
else,
  fpout = fopen(outfilename,'w');
end;

% read in the header, except for the number of frames
header = fread(fpin, header_size - 8, datatype);

% write the header
fwrite(fpout,header);

% write the number of frames
fwrite(fpout,nframes,'integer*8');

% copy some frames
fprintf('Copying ...\n');
for i = 1:nframes,

  % go to the next frame
  if i == 1,
    if startframe > 1,
      fseek(fpin,(startframe-1)*bytes_per_chunk,'cof');
    end;
  else,
    if nskip > 1,
      fseek(fpin,(nskip-1)*bytes_per_chunk,'cof')
    end;
  end;

  % read in the frame
  buf = fread(fpin,bytes_per_chunk,'uint8');

  % write out the frame
  fwrite(fpout,buf);

end;

% close the files
fclose(fpin);
fclose(fpout);
