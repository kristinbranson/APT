classdef MovieIndexSetFixed < MovieIndexSet
  properties
    movs
  end
  methods
    function obj = MovieIndexSetFixed(iMovs)
      obj.movs = iMovs(:)';
    end
    function str = getPrettyString(obj)
      str = sprintf('Movies: %s',mat2str(obj.movs));
    end
    function iMovs = getMovieIndices(obj,lObj)
      iMovs = obj.movs;
    end
  end
end