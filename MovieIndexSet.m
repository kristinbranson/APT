classdef MovieIndexSet < handle
  % A MovieIndexSet represents a conceptual set of movies for an APT 
  % project. An explicit list of movie(set) indices can be generated given 
  % a labeler instance.

  enumeration
    AllMov ('All movies')
    CurrentMov ('Current movie')
    SelMov ('Selected movies')
  end

  properties
    prettyString
  end
  methods
    function obj = MovieIndexSet(ps)
      obj.prettyString = ps;
    end
    function iMovs = getMovieIndices(obj,labelerObj)
      if ~labelerObj.hasMovie
        iMovs = zeros(0,1);
      else
        switch obj
          case MovieIndexSet.CurrentMov
            iMovs = labelerObj.currMovie;
          case MovieIndexSet.SelMov
            iMovs = labelerObj.moviesSelected;            
          case MovieIndexSet.AllMov
            iMovs = 1:labelerObj.nmovies;
          otherwise
            assert(false,'Unknown MovieIndexSet.');
        end
      end
    end
  end
end
