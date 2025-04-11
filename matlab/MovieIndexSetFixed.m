classdef MovieIndexSetFixed < MovieIndexSet
  properties
    mIdxs
  end
  methods
    function obj = MovieIndexSetFixed(mi)
      assert(isa(mi,'MovieIndex'));
      obj.mIdxs = mi(:)';
    end
    function str = getPrettyString(obj)
      mi = obj.mIdxs;
      [tf,gt] = mi.isConsistentSet;
      if tf
        if gt
          str = sprintf('Movies (gt): %s',mat2str(abs(mi)));
        else
          str = sprintf('Movies: %s',mat2str(abs(mi)));
        end
      else
        str = sprintf('Movies (mixed GT/reg): %s',mat2str(int32(mi)));
      end
    end
    function mIdx = getMovieIndices(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;      
      mIdx = obj.mIdxs;
    end
  end
end