function [data, stamps, f] = fmf_read( filename, f_start, nframes, incr, show_progress )
% [data, stamps, f] = fmf_read( filename, f_start, nframes, incr, show_progress )
%
% reads FlyMovieFormat data from FILENAME
%
% starts with frame F_START (1-based) [defaults to 1], reads NFRAMES [defaults to inf]
%
% DATA is HxWxNFRAMES where H and W are the height and width of a
% frame STAMPS are the timestamps for each frame, double format F is
% the number of frames actually read
%
% JAB 6/30/04

CHUNKSIZE = 200;

if nargin < 5, show_progress = 0; end
if nargin < 4, incr = 1; end;
if nargin < 3, nframes = inf; end
if nargin < 2, f_start = 1; end

[header_size, version, h, w, bytes_per_chunk, max_n_frames, data_format] = ...
    fmf_read_header( filename );
if( nframes ~= inf && (nframes-1)*incr + f_start > max_n_frames ),
  nframes = floor( (max_n_frames - f_start)/incr + 1);
  warning( sprintf( 'not enough frames in file -- returning %d frames', nframes ) );
elseif nframes == inf,
  nframes = max_n_frames - f_start + 1;
end

datatype = fmf_get_datatype(data_format);
if isinf(nframes),
  % allocate one chunk at a time
  data = zeros( h, w, CHUNKSIZE, datatype );
  stamps = zeros(1,CHUNKSIZE);
else,
  data = zeros( h, w, nframes, datatype );
  stamps = zeros( 1, nframes );
end;
% read frames
fp = fopen( filename, 'r' );
fseek( fp, header_size, 'bof' );
% skip the first f_start - 1 frames
fseek( fp, bytes_per_chunk*(f_start-1), 'cof' );
% add wait bar
if show_progress
 bar_handle = waitbar(0,'Please wait...');
end
seekfailed = 0;
for f=1:nframes,
  if show_progress & ~isinf(nframes),
    waitbar(f/nframes, bar_handle)
  end
  % do we need to allocate more memory?
  if f > size(data,3),
    data = cat(3,data,zeros(h,w,CHUNKSIZE,datatype));
    stamps = cat(2,stamps,zeros(1,CHUNKSIZE));
  end;
  
  [data(:,:,f), stamps(f)] = fmf_read_frame( fp, h, w, ...
					     bytes_per_chunk, data_format );
  % were we unable to read frame f?
  if stamps(f) == 9e9,
    warning( sprintf( 'not enough frames in file -- returning %d frames', f-1 ) );
    % return data we've read up until now
    data = data(:,:,1:f-1);
    stamps = stamps(1:f-1);
    break;
  end;      

  % skip incr-1 frames
  if f < nframes,
    seekfailed = fseek(fp,(incr-1)*bytes_per_chunk,'cof');
    % stop seeking when we can't seek anymore
    if seekfailed ~= 0,
      break;
    end;
  end;
  % if we've reached the end of the file, return the data we've
  % read up until now. 
  if seekfailed~=0,
    data = data(:,:,1:f);
    stamps = stamps(1:f);
    break;
  end;
end
fclose( fp );

if show_progress & ~isinf(nframes),
  close(bar_handle)
end