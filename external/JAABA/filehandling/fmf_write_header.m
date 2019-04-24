function fp = fmf_write_header( filename, header_size, version, ...
                                f_height, f_width, bytes_per_chunk, ...
                                max_n_frames, data_format)
% fp = fmf_write_header( filename, header_size, version, 
%                        f_height, f_width, bytes_per_chunk, 
%                        max_n_frames, data_format )
%
% writes FlyMovieFormat header data to FILENAME and returns a file
% pointer FP.
%
% VERSION is the fmf version
% HEADER_SIZE is in bytes for the appropriate file format (fmf VERSION)
% F_HEIGHT and F_WIDTH are the number of pixels in a frame
% BYTES_PER_CHUNK is the number of bytes per frame in the file
% MAX_N_FRAMES is the number of frames in the file
% DATA_FORMAT is the format in which each pixel is stored. 
%
% KMB 04/07

fp = fopen( filename, 'w' );
if fp <= 0,
  error(['Could not open file ',filename,' for writing.']);
end;

% write header
if version ~= 1 & version ~= 3, 
  error( 'version not supported -- FMF versions 1 and 3 only' );
end;

fwrite(fp,version,'uint32');
if version == 1,
  data_format = 'MONO8';
  bits_per_pixel = 8;
end
if version == 3,
  fwrite(fp,length(data_format),'uint32');
  fwrite(fp,data_format);
  bits_per_pixel = fmf_get_bits_per_pixel(data_format);
  fwrite(fp,bits_per_pixel,'uint32');
end

fwrite(fp,f_height,'uint32');
fwrite(fp,f_width,'uint32');
fwrite(fp,bytes_per_chunk,'integer*8');
fwrite(fp,max_n_frames,'integer*8');
curr = ftell(fp);
if header_size > curr,
  fwrite(fp,zeros(header_size-curr,1), 'uint8');
end;