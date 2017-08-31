classdef MovieSet
  properties
    prettyString
  end
  methods
    function obj = MovieSet(ps)
      obj.prettyString = ps;
    end
%     function str = menuStr(obj,labelerObj)
%       % Create pretty-string for UI
%       lprop = obj.labelerProp;
%       if isempty(lprop)
%         str = obj.prettyStringPat;
%       else
%         val = labelerObj.(lprop);
%         str = sprintf(obj.prettyStringPat,val);
%       end
% %     end
    function iMovs = getMovieIndices(obj,labelerObj)
      if ~labelerObj.hasMovie
        iMovs = zeros(0,1);
      else
        switch obj
          case MovieSet.CurrMov
            iMovs = labelerObj.currMovie;
          case MovieSet.SelMov
            iMovs = labelerObj.moviesSelected;            
          case MovieSet.AllMov
            iMovs = 1:labelerObj.nmoviesGTaware;
          otherwise
            assert(false,'Unknown movieset.');
        end
      end
    end
  end
  enumeration
    CurrentMov ('Current movie')
    SelMov ('Selected movies')
    AllMov ('All movies')
    %AllMovTrack ('All movies with tracking')
  end
end
    
    