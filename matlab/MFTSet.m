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
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      
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
    
    function str = getPrettyStrCompact(obj,labelerObj)
      % Create shorter pretty-string for UI, iss #161
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      
      movstr = obj.movieIndexSet.getPrettyString;
      if ~strcmpi(movstr,'current movie')
        str = obj.getPrettyStr(labelerObj);
        return;
      end
      
      frmstr = lower(obj.frameSet.getPrettyString(labelerObj));
      [decstr,decval] = obj.decimation.getPrettyString(labelerObj);
      decstr = lower(decstr);
      if labelerObj.nTargets>1
        tgtstr = lower(obj.targetSet.getPrettyString(labelerObj));
        if decval==1
          str = sprintf('%s, %s',tgtstr,frmstr);
        else
          str = sprintf('%s, %s, %s',tgtstr,frmstr,decstr);
        end
      else
        if decval==1
          str = frmstr;
        else
          str = sprintf('%s, %s',frmstr,decstr);
        end
      end
      str(1) = upper(str(1));
    end
    
    function str = getPrettyStrMoreCompact(obj,labelerObj)
      % Create shorter pretty-string for UI, iss #161
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      
      movstr = obj.movieIndexSet.getPrettyString;
      if ~strcmpi(movstr,'current movie')
        str = obj.getPrettyStr(labelerObj);
        return;
      end
      
      frmstr = lower(obj.frameSet.getPrettyCompactString(labelerObj));
      [decstr,decval] = obj.decimation.getPrettyCompactString(labelerObj);
      decstr = lower(decstr);
      if labelerObj.nTargets>1
        tgtstr = lower(obj.targetSet.getPrettyCompactString(labelerObj));
        if decval==1
          str = sprintf('%s, %s',tgtstr,frmstr);
        else
          str = sprintf('%s, %s, %s',tgtstr,frmstr,decstr);
        end
      else
        if decval==1
          str = frmstr;
        else
          str = sprintf('%s, %s',frmstr,decstr);
        end
      end
      str(1) = upper(str(1));
    end  % function
    
    function tblMFT = getMFTable(obj,labelerObj,varargin)
      % tblMFT: MFTable with MFTable.ID
      %
      % tblMFT is computed via the following procedure.
      % 1. List of movie indices is found by querying .movieIndexSet.
      % 2. For each movie, the list of targets is found by querying
      %   .targetSet.
      % 3. The decimation is found by querying .decimation.
      % 4. For each movie/target/decimation, the list of frames is found by
      % querying .frameSet.
      %
      % For MA track, supply istrack=true to varargin. For MA projects,
      % 'targets' can refer to either distinct label IDs, or tracklets.
      % When tracking, neither currently applies, as entire frames are
      % processed.

      [wbObj,istrack] = myparse(varargin,...
        'wbObj',[], ... % (opt) WaitBarWithCancel or ProgressMeter. If cancel, tblMFT indeterminate.
        'istrack',false ... % if true and labelerObj is MA, then 'targets' are ignored/meaningless.
        ); 
      tfWB = ~isempty(wbObj);      
      
      if ~labelerObj.hasMovie
        mov = MovieIndex(zeros(0,1));
        frm = zeros(0,1);
        iTgt = zeros(0,1);
        tblMFT = table(mov,frm,iTgt);
      else
        mis = obj.movieIndexSet.getMovieIndices(labelerObj.getMftInfoStruct());
        decFac = obj.decimation.getDecimation(labelerObj.getMftInfoStruct());
        tgtSet = obj.targetSet;
        frmSet = obj.frameSet;
        
        nMovs = numel(mis);
        tblMFT = cell(0,1);
        isMA = labelerObj.maIsMA;
        if isMA && istrack
          % nan targets here represent "Any target/MA" and should be so
          % interpreted within frmSet.
          iTgtsArr = repmat({nan},nMovs,1);          
        elseif isMA || labelerObj.hasTrx
          iTgtsArr = tgtSet.getTargetIndices(labelerObj,mis);
        else
          iTgtsArr = repmat({1},nMovs,1);
        end
        
        if tfWB
          if isa(wbObj, 'ProgressMeter') ,
            wbObj.start('message', 'Fetching table', 'denominator', nMovs) ;
            oc = onCleanup(@()wbObj.finish());
          else
            wbObj.startPeriod('Fetching table','shownumden',true,'denominator',nMovs);
            oc = onCleanup(@()wbObj.endPeriod());
          end
        end
        for i=1:nMovs
          if tfWB
            if isa(wbObj, 'ProgressMeter') ,
              if wbObj.wasCanceled
                % tblMFT is raw/cell, return it anyway
                return
              end
              wbObj.bump(i);
            else
              if wbObj.isCancel
                % tblMFT is raw/cell, return it anyway
                return
              end
              wbObj.updateFracWithNumDen(i);
            end              
          end
          
          mIdx = mis(i);
          iTgts = iTgtsArr{i};
          for j=1:numel(iTgts)
            iTgt = iTgts(j);
            frms = frmSet.getFrames(labelerObj,mIdx,iTgt,decFac);
            nfrm = numel(frms);
            tblMFT{end+1,1} = table(...
              repmat(mIdx,nfrm,1),frms(:),repmat(iTgt,nfrm,1),...
              'VariableNames',{'mov' 'frm' 'iTgt'}); %#ok<AGROW>
          end
        end
        tblMFT = cat(1,tblMFT{:});
      end  % if ~labelerObj.hasMovie
    end  % function    
  end  % methods  
end  % classdef
    