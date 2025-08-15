classdef ShiftArrowMovieNavMode
  enumeration
    NEXTTIMELINE ('Next shown in timeline')
    NEXTTIMELINETHRESH ('Next where timeline stat exceeds threshold')
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
    function [tffound,f] = seekFrame(obj,lObj,dir,thresh,cmp)
      % dir: +1 or -1
      % 
      % f: frame
      
      f0 = lObj.currFrame;
      switch obj
        case ShiftArrowMovieNavMode.NEXTTIMELINE
          tldata = lObj.infoTimelineModel.tldata;
          [tffound,f] = Labels.seekSmallLpos(tldata,f0,dir);
        case ShiftArrowMovieNavMode.NEXTTIMELINETHRESH
          tldata = lObj.infoTimelineModel.tldata;
          [tffound,f] = Labels.seekSmallLposThresh(tldata,f0,dir,thresh,cmp);
        case ShiftArrowMovieNavMode.NEXTLABELED
          %lpos = lObj.labeledposCurrMovie;
          s = lObj.labelsCurrMovie;
          if lObj.hasTrx,
            tgt = lObj.currTarget;
          else
            tgt = [];
          end
          [tffound,f] = Labels.findLabelNear(s,f0,tgt,dir*2);
        case ShiftArrowMovieNavMode.NEXTIMPORTED
          if lObj.gtIsGTMode
            warningNoTrace('No imported labels available in GT mode.');
            tffound = false;
            f = nan;
          else
            iMov = lObj.currMovie;
            %lpos = lObj.labeledpos2{iMov};
            s = lObj.labels2{iMov};
            [tffound,f] = Labels.findLabelNear(s,f0,lObj.currTarget,dir*2);
            %[tffound,f] = Labels.seekBigLpos(lpos,f0,dir,lObj.currTarget);
          end
        case ShiftArrowMovieNavMode.NEXTTRACKED
          tObj = lObj.tracker;
          if isempty(tObj)
            warningNoTrace('This project does not have a tracker.');
            tffound = false;
            f = nan;
          else
            [tffound,xy,~,sf,~] = tObj.getTrackingResultsCurrMovieTgt();
            if tffound
              off = 1-sf;
              f0mod = f0+off;
              [tffound,f] = Labels.seekBigLpos(xy,f0mod,dir);
              f = f-off;
            else
              f = nan;
            end
          end
        otherwise
          assert(false);
      end
    end
  end
  
end