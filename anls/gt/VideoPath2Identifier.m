function mstr = VideoPath2Identifier(moviename,datatype)

switch datatype,
  case 'stephen'
    mstr = StephenVideo2Identifier(moviename);
  case {'jan','roian'}
    [~,mstr] = fileparts(moviename);
  case 'jay',
    ss = strsplit(moviename,'/');
    if strcmp(ss{end},'movie_comb.avi'),
      mstr = ss{end-1};
    else
      mstr = ss{end}(1:end-4);
    end
  otherwise
    error('not implemented');
end
