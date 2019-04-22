function v = isglobalpath(path)

v = (~ispc && ~isempty(path) && path(1) == '/') || ...
  (ispc && ~isempty(regexp(path,'^[A-Z]:[\\/]','once')));
            