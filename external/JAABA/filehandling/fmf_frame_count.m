function f = fmf_frame_count( filename )
% frames = fmf_frame_count( filename )
%
% counts the number of frames in a FlyMovieFormat file
% and writes it into the file header if it's not already there
%
% JAB 6/30/04

fp = fopen( filename, 'r' );

version = double( fread( fp, 1, 'uint32' ) );
if version ~= 1,
    error( 'version not supported -- FMF version 1 only' );
end
fseek( fp, 20, 'bof' );
max_n_frames = double( fread( fp, 1, 'long' ) );

if max_n_frames ~= 0,
	f = max_n_frames;
	return
end

% count frames
fseek( fp, 12, 'bof' );
frame_size = double( fread( fp, 1, 'long' ) );
fseek( fp, 8, 'cof' );
f = 0;
while 1,
	err = fseek( fp, frame_size, 'cof' );
	if err == -1, break, end
	f = f + 1;
end

fclose( fp );

% update frame count in file
return % update not working!!
fp = fopen( filename, 'r+' );
fseek( fp, 20, 'bof' );
fwrite( fp, f, 'long' );
fclose( fp );
