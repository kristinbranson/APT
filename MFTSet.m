classdef MFTSet < handle
  % An MFTSet is a set of movies/frames/targets in an APT project.
  %
  % Conceptually, it is the 4-way cross product of i) a movie
  % specification (eg "current movie"), ii) a frameset specification (eg
  % "frames 1:100", iii) a target specification (eg "all targets") and iv)
  % a decimation factor (eg 2->"every other frame").
    
  properties
    movieIndexSet % scalar MovieIndexSet
    frameSet % scalar FrameSet
    decimation % scale FrameDecimation
    targetSet % scalar TargetSet
  end
  
  methods
  
    function obj = MFTSet(mset,fset,dec,tset)
      assert(isa(mset,'MovieIndexSet'));
      assert(isa(fset,'FrameSet'));
      assert(isa(dec,'FrameDecimation'));
      assert(isa(tset,'TargetSet'));
      
      obj.movieIndexSet = mset;
      obj.frameSet = fset;
      obj.decimation = dec;
      obj.targetSet = tset;
    end
    
    function str = getPrettyStr(obj,labelerObj)
      % Create pretty-string for UI
      
      movstr = obj.movieIndexSet.getPrettyString;
      frmstr = lower(obj.frameSet.getPrettyString(labelerObj));
      [decstr,decval] = obj.decimation.getPrettyString(labelerObj);
      decstr = lower(decstr);
      if labelerObj.nTargets>1
        tgtstr = lower(obj.targetSet.getPrettyString(labelerObj));
        if strcmpi(movstr,'current movie') && strcmpi(tgtstr,'current target')
          movtgtstr = 'current movie/target';
        else
          movtgtstr = sprintf('%s, %s',movstr,tgtstr);
        end
        if decval==1
          str = sprintf('%s, %s',movtgtstr,frmstr);
        else
          str = sprintf('%s, %s, %s',movtgtstr,frmstr,decstr);
        end
      else
        if decval==1
          str = sprintf('%s, %s',movstr,frmstr);
        else
          str = sprintf('%s, %s, %s',movstr,frmstr,decstr);
        end
      end
      str(1) = upper(str(1));
    end
    
    function tblMFT = getMFTable(obj,labelerObj)
      % tblMFT: MFTable with MFTable.ID
      
      if ~labelerObj.hasMovie
        mov = zeros(0,1);
        frm = zeros(0,1);
        iTgt = zeros(0,1);
        tblMFT = table(mov,frm,iTgt);
      else
        iMovs = obj.movieIndexSet.getMovieIndices(labelerObj);
        decFac = obj.decimation.getDecimation(labelerObj);
        tgtSet = obj.targetSet;
        frmSet = obj.frameSet;
        
        nMovs = numel(iMovs);
        tblMFT = cell(0,1);
        for i=1:nMovs
          iMov = iMovs(i);
          iTgts = tgtSet.getTargetIndices(labelerObj,iMov);
          for j=1:numel(iTgts)
            iTgt = iTgts(j);
            frms = frmSet.getFrames(labelerObj,iMov,iTgt,decFac);
            nfrm = numel(frms);
            tblMFT{end+1,1} = table(...
              repmat(iMov,nfrm,1),frms(:),repmat(iTgt,nfrm,1),...
              'VariableNames',{'mov' 'frm' 'iTgt'}); %#ok<AGROW>
          end
        end
        tblMFT = cat(1,tblMFT{:});
      end
    end
    
  end 
 
end
    