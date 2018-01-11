classdef MovieIndexSet < handle
  methods (Abstract)
    str = getPrettyString(obj)
    
    % mIdx: MovieIndex vector
    mIdx = getMovieIndices(obj,labelerObj)
  end
end