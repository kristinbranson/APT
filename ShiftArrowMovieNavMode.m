classdef ShiftArrowMovieNavMode
  enumeration
    NEXTTIMELINE ('Next shown in timeline')
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
    function [tffound,f] = seekFrame(obj,lObj,dir)
      % dir: +1 or -1
      % 
      % f: frame
      
      f0 = lObj.currFrame;
      switch obj
        case ShiftArrowMovieNavMode.NEXTTIMELINE
          tldata = lObj.gdata.labelTLInfo.tldata;
          [tffound,f] = Labeler.seekSmallLpos(tldata,f0,dir);
        case ShiftArrowMovieNavMode.NEXTLABELED
          lpos = lObj.labeledposCurrMovie;
          [tffound,f] = Labeler.seekBigLpos(lpos,f0,dir,lObj.currTarget);
        case ShiftArrowMovieNavMode.NEXTIMPORTED
          if lObj.gtIsGTMode
            warningNoTrace('No imported labels available in GT mode.');
            tffound = false;
            f = nan;
          else
            iMov = lObj.currMovie;
            lpos = lObj.labeledpos2{iMov};
            [tffound,f] = Labeler.seekBigLpos(lpos,f0,dir,lObj.currTarget);
          end
        case ShiftArrowMovieNavMode.NEXTTRACKED
          tObj = lObj.tracker;
          if isempty(tObj)
            warningNoTrace('This project does not have a tracker.');
            tffound = false;
            f = nan;
          else
            lpos = tObj.xyPrdCurrMovie;
            [tffound,f] = Labeler.seekBigLpos(lpos,f0,dir,lObj.currTarget);
          end
        otherwise
          assert(false);
      end
    end
  end
  
end