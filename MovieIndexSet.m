classdef MovieIndexSet < handle
  methods (Abstract)
    str = getPrettyString(obj)
    iMovs = getMovieIndices(obj,labelerObj)
  end
end