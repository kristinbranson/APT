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
    AllMov = MovieIndexSetVariable('All movies',@lclAllMoviesGetMovieIndexHook); % "movies per current GT mode"
    CurrMov = MovieIndexSetVariable('Current movie',@lclCurrMovieGetMovieIndexHook);
    SelMov = MovieIndexSetVariable('Selected movies',@lclSelMovieGetMovieIndexHook);
    AllGTMov = MovieIndexSetVariable('All GT movies',@lclAllGTMoviesGetMovieIndexHook);
  end  
end

function mIdx = lclAllMoviesGetMovieIndexHook(lObj)
nmov = lObj.nmoviesGTaware;
mIdx = MovieIndex(1:nmov,lObj.gtIsGTMode);
end
function mIdx = lclCurrMovieGetMovieIndexHook(lObj)
mIdx = lObj.currMovIdx;
end
function mIdx = lclSelMovieGetMovieIndexHook(lObj)
if lObj.gtIsGTMode
  error('Unsupported in GT mode.');
end
mIdx = MovieIndex(lObj.moviesSelected);
end
function mIdx = lclAllGTMoviesGetMovieIndexHook(lObj)
iMovs = 1:lObj.nmoviesGT;
mIdx = MovieIndex(-iMovs);
end