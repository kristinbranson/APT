classdef MovieIndexSetVariable < MovieIndexSet
  properties
    prettyString
    getMovieIndicesHook % fcn with sig iMovs = fcn(labeler)
  end
  methods
    function obj = MovieIndexSetVariable(ps,fcn)
      obj.prettyString = ps;
      obj.getMovieIndicesHook = fcn;
    end
    function str = getPrettyString(obj)
      str = obj.prettyString;
    end
    function iMovs = getMovieIndices(obj,lObj)
      if ~lObj.hasMovie
        iMovs = zeros(1,0);
      else
        iMovs = obj.getMovieIndicesHook(lObj);
        iMovs = iMovs(:)';
      end
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllMov = MovieIndexSetVariable('All movies',@(lo)1:lo.nmovies);
    CurrMov = MovieIndexSetVariable('Current movie',@(lo)lo.currMovie);
    SelMov = MovieIndexSetVariable('Selected movies',@(lo)lo.moviesSelected);
  end  
end