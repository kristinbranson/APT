function [header_size, version, f_height, f_width, bytes_per_chunk, ...
          max_n_frames, data_format] = fmf_read_header( filename )
% [header_size,version,nr,nc,bytes_per_chunk,nframes,data_format] = fmf_read_header(filename)
%
% reads FlyMovieFormat header data from FILENAME
%
% HEADER_SIZE is in bytes for the appropriate file format (fmf VERSION)
% F_HEIGHT and F_WIDTH are the number of pixels in a frame
% BYTES_PER_CHUNK is the number of bytes per frame in the file
% MAX_N_FRAMES is the number of frames in the file
%
% JAB 7/1/04
%
% Fixed bug in reading number of frames. 
% 
% KMB 05/22/07

header_size = 28;

fp = fopen( filename, 'r' );

% read header
version = double( fread( fp, 1, 'uint32' ) );
if (version ~= 1) && (version ~= 3),
  error( 'version not supported -- FMF versions 1 and 3 only' );
end

if version == 1,
  data_format = 'MONO8';
  bits_per_pixel = 8;
end

if version == 3,
  format_len = double( fread( fp, 1, 'uint32' ) );
  data_format = strcat( char( fread( fp, format_len))');
  bits_per_pixel = double( fread( fp, 1, 'uint32' ) );
  header_size = header_size + format_len + 8;
end

f_height = double( fread( fp, 1, 'uint32' ) );
f_width = double( fread( fp, 1, 'uint32' ) );
bytes_per_chunk = double( fread( fp, 1, 'uint64' ) );
frame_count_location = ftell(fp);
max_n_frames = double( fread( fp, 1, 'uint64' ) );

if max_n_frames <= 0,
  fprintf('Resorting to computing number of frames from size of file.\n');
  pcurr = ftell(fp);
  fseek(fp,0,'eof');
  nbytestot = ftell(fp);
  fseek(fp,pcurr,'bof');
  max_n_frames = floor((nbytestot - header_size) / bytes_per_chunk);
end

fclose( fp );
