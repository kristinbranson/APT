function [stamps, f] = fmf_read_timestamps( filename, f_start, nframes, incr, show_progress )
% [stamps, f] = fmf_read_timestamps( filename, f_start, nframes, incr, show_progress )
%
% reads FlyMovieFormat data from FILENAME
%
% starts with frame F_START (1-based) [defaults to 1], reads NFRAMES [defaults to inf]
%

CHUNKSIZE = 200;

if nargin < 5, show_progress = 0; end
if nargin < 4, incr = 1; end;
if nargin < 3, nframes = inf; end
if nargin < 2, f_start = 1; end

[header_size, version, h, w, bytes_per_chunk, max_n_frames, data_format] = ...
    fmf_read_header( filename ); %#ok<NASGU,ASGLU>
if( nframes ~= inf && (nframes-1)*incr + f_start > max_n_frames ),
  nframes = floor( (max_n_frames - f_start)/incr + 1);
  warning( 'not enough frames in file -- returning %d frames', nframes );
elseif nframes == inf,
  nframes = max_n_frames - f_start + 1;
end

if isinf(nframes),
  % allocate one chunk at a time
  stamps = zeros(1,CHUNKSIZE);
else
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
  if show_progress && ~isinf(nframes),
    waitbar(f/nframes, bar_handle)
  end
  % do we need to allocate more memory?
  if f > length(stamps),
    stamps = cat(2,stamps,zeros(1,CHUNKSIZE));
  end;
  if feof( fp ),
    warning( 'not enough frames in file -- returning %d frames', f-1 );
    stamps = stamps(1:f-1);
    break;
  else
    stamps(f) = fread( fp, 1, 'double' );
  end

  % skip incr-1 frames
  if f < nframes,
    seekfailed = fseek(fp,incr*bytes_per_chunk-8,'cof');
    % stop seeking when we can't seek anymore
    if seekfailed ~= 0,
      break;
    end;
  end;
  % if we've reached the end of the file, return the data we've
  % read up until now. 
  if seekfailed~=0,
    stamps = stamps(1:f);
    break;
  end;
end
fclose( fp );

if show_progress && ~isinf(nframes),
  close(bar_handle)
end