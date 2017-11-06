classdef ShiftArrowMovieNavMode
  enumeration
    NEXTLABELED ('Next labeled')
    NEXTTRACKED ('Next tracked')
    NEXTIMPORTED ('Next imported')
  end
  properties 
    prettyStr
  end
  methods
    function obj = ShiftArrowMovieNavMode(str)
      obj.prettyStr = str;
    end
    function lpos = getLposSeek(obj,lObj)
      switch obj
        case ShiftArrowMovieNavMode.NEXTLABELED
          lpos = lObj.labeledposCurrMovie;
        case ShiftArrowMovieNavMode.NEXTIMPORTED
          if lObj.gtIsGTMode
            error('Unsupported in GT mode.');
          end
          iMov = lObj.currMovie;
          lpos = lObj.labeledpos2{iMov};
        case ShiftArrowMovieNavMode.NEXTTRACKED
          tObj = lObj.tracker;
          if isempty(tObj)
            error('This project does not have a tracker.');
          end
          lpos = tObj.xyPrdCurrMovie;
        otherwise
          assert(false);
      end
    end
  end
end