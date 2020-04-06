function nframes = GetMovieNFrames(varargin)

nframes = zeros(1,nargin);
for i = 1:nargin,
  moviefile = varargin{i};
  if ~exist(moviefile,'file'),
    fprintf('0');
  else
    [~,nframes(i),fid] = get_readframe_fcn(moviefile);
    fprintf('%d',nframes(i));
  end
  if fid > 0,
    fclose(fid);
  end
  if i == nargin,
    fprintf('\n');
  else
    fprintf(',');
  end
end