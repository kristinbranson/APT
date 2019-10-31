classdef MovieIndexSetVariable < MovieIndexSet
  properties
    prettyString
    prettyCompactString
    getMovieIndicesHook % fcn with sig mIdx = fcn(labeler)
  end
  methods
    function obj = MovieIndexSetVariable(ps,cps,fcn)
      obj.prettyString = ps;
      obj.prettyCompactString = cps;
      obj.getMovieIndicesHook = fcn;
    end
    function str = getPrettyString(obj)
      str = obj.prettyString;
    end
    function str = getPrettyCompactString(obj)
      str = obj.prettyCompactString;
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
    AllMov = MovieIndexSetVariable('All movies','All mov',@lclAllMoviesGetMovieIndexHook); % "movies per current GT mode"
    CurrMov = MovieIndexSetVariable('Current movie','Cur mov',@lclCurrMovieGetMovieIndexHook);
    SelMov = MovieIndexSetVariable('Selected movies','Sel mov',@lclSelMovieGetMovieIndexHook);
    AllTrnMov = MovieIndexSetVariable('All Training movies','Trn mov',@lclAllNonGTMoviesGetMovieIndexHook);
    AllGTMov = MovieIndexSetVariable('All GT movies','GT mov',@lclAllGTMoviesGetMovieIndexHook);
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
mIdx = lObj.moviesSelected;
end
function mIdx = lclAllNonGTMoviesGetMovieIndexHook(lObj)
iMovs = 1:lObj.nmovies;
mIdx = MovieIndex(iMovs);
end
function mIdx = lclAllGTMoviesGetMovieIndexHook(lObj)
iMovs = 1:lObj.nmoviesGT;
mIdx = MovieIndex(-iMovs);
end