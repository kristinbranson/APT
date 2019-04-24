function fmf_animate( filename, f_start, nframes, incr, flip, cm, aoi )
% fmf_animate( filename, f_start, nframes, incr, flip, cm, aoi )
%
% F_START is the starting frame (1-based) [default 1]
% NFRAMES is the number of frames to animate [default inf]
% INCR is the frame index increment to plot [default 1]
% FLIP is a flag to flip the image left-to-right [default 0]
% CM is an optional colormap [default gray]
%   help graph3d to see colormap possibilities
% AOI is an optional area of interest, a vector containing x,y,w,h [default full image]
%
% JAB 7/1/04

%    hsv        - Hue-saturation-value color map.
%    hot        - Black-red-yellow-white color map.
%    gray       - Linear gray-scale color map.
%    bone       - Gray-scale with tinge of blue color map.
%    copper     - Linear copper-tone color map.
%    pink       - Pastel shades of pink color map.
%    white      - All white color map.
%    flag       - Alternating red, white, blue, and black color map.
%    lines      - Color map with the line colors.
%    colorcube  - Enhanced color-cube color map.
%    vga        - Windows colormap for 16 colors.
%    jet        - Variant of HSV.
%    prism      - Prism color map.
%    cool       - Shades of cyan and magenta color map.
%    autumn     - Shades of red and yellow color map.
%    spring     - Shades of magenta and yellow color map.
%    winter     - Shades of blue and green color map.
%    summer     - Shades of green and yellow color map.
if nargin < 6, cm = 'gray'; end
if nargin < 5, flip = 0; end
if nargin < 4, incr = 1; end
if nargin < 3, nframes = inf; end
if nargin < 2, f_start = 1; end

[header_size, version, h, w, frame_size, max_n_frames, data_format] = fmf_read_header( filename );
if( nframes ~= inf && nframes + f_start - 1 > max_n_frames ),
	nframes = max_n_frames - f_start + 1;
	warning( 'fmf_short', sprintf( 'not enough frames in file -- returning %d frames', nframes ) );
elseif nframes == inf,
	nframes = max_n_frames - f_start + 1;
end

data = uint8( zeros( h, w ) );
figure( 19 ); clf
pause( 1e-3 )

% read frames
fp = fopen( filename, 'r' );
fseek( fp, header_size, 'bof' );
fseek( fp, frame_size*(f_start-1), 'cof' );
for f=1:incr:nframes,
	for i=1:incr, data = fmf_read_frame( fp, h, w, frame_size, data_format ); end

	% flip
	if flip, data = fliplr( data ); end
	
	% truncate to area of interest
	if nargin == 7,
		data = data(aoi(2):aoi(2)+aoi(4)-1,aoi(1):aoi(1)+aoi(3)-1);
	end

	% for first frame, size figure to real size
	if f == 1,
		fprintf( 1, 'frame:           ' );
		imshow( data )
		eval( sprintf( 'colormap %s', cm ) )
	end

	% display image
	fprintf( 1, '\b\b\b\b\b\b\b\b\b\b\b\b\b%6d/%6d', f+f_start-1, nframes+f_start-1 );
	%imagesc( data )
	image( data )
	set( gca, 'visible', 'off' )
	pause( 1e-3 ) % allow plotting to finish
end
fclose( fp );
fprintf( 1, '\n' );
beep
