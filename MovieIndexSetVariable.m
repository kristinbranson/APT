classdef MovieIndexSetVariable < MovieIndexSet
  properties
    prettyString
    getMovieIndicesHook % fcn with sig mIdx = fcn(labeler)
  end
  methods
    function obj = MovieIndexSetVariable(ps,fcn)
      obj.prettyString = ps;
      obj.getMovieIndicesHook = fcn;
    end
    function str = getPrettyString(obj)
      str = obj.prettyString;
    end
    function mIdx = getMovieIndices(obj,lObj)
      if ~lObj.hasMovie
        mIdx = MovieIndex(zeros(1,0));
      else
        mIdx = obj.getMovieIndicesHook(lObj);
        mIdx = mIdx(:)';
      end
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllMov = MovieIndexSetVariable('All movies',@(lo)MovieIndex(1:lo.nmovies)); % "All regular movies"
    CurrMov = MovieIndexSetVariable('Current movie',@lclCurrMovieGetMovieIndexHook);
    SelMov = MovieIndexSetVariable('Selected movies',@lclSelMovieGetMovieIndexHook);
  end  
end

function mIdx = lclCurrMovieGetMovieIndexHook(lObj)
assert(~lObj.gtIsGTMode);
mIdx = MovieIndex(lObj.currMovie);
end

function mIdx = lclSelMovieGetMovieIndexHook(lObj)
assert(~lObj.gtIsGTMode);
mIdx = MovieIndex(lObj.moviesSelected);
end