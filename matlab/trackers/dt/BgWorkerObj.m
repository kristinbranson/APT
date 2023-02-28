classdef BgWorkerObj < handle
  % Object deep copied onto BG worker. To be used with
  % BGWorkerContinuous
  % 
  % Responsibilities:
  % - Poll filesystem for updates
  % - Be able to read/parse the current state on disk
  
  % Class diagram 20191223
  % Only leaf/concrete classes Bg{Train,Track}WorkerObj{BEtype} are 
  % instantiated.
  % 
  % BgWorkerObj
  % BgTrainWorkerObj < BgWorkerObj
  % BgTrackWorkerObj < BgWorkerObj
  % BgWorkerObjLocalFilesys < BgWorkerObj
  %   BgWorkerObjDocker < BgWorkerObjLocalFilesys  
  %   BgWorkerObjBsub < BgWorkerObjLocalFilesys
  %   BgWorkerObjConda < BgWorkerObjLocalFilesys  
  % BgWorkerObjAWS < BgWorkerObj
  %
  % Train concrete classes
  % BgTrainWorkerObjDocker < BgWorkerObjDocker & BgTrainWorkerObj  
  % BgTrainWorkerObjConda < BgWorkerObjConda & BgTrainWorkerObj
  % BgTrainWorkerObjBsub < BgWorkerObjBsub & BgTrainWorkerObj
  % BgTrainWorkerObjAWS < BgWorkerObjAWS & BgTrainWorkerObj
  %
  % Track concrete classes
  % BgTrackWorkerObjDocker < BgWorkerObjDocker & BgTrackWorkerObj  
  % BgTrackWorkerObjConda < BgWorkerObjConda & BgTrackWorkerObj  
  % BgTrackWorkerObjBsub < BgWorkerObjBsub & BgTrackWorkerObj  
  % BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  
  properties
    % TODO Reconcile/cleanup
    % For BgTrainWorkerObjs
    %   In general, for training, the number of actual views is not
    %   important to BgWorkers; the concept of "views" can be replaced by 
    %   "stages" with no changes in code.
    % For BgTrackWorkerObjs, nviews is still the number of actual views,
    %   and may differ from numel(dmcs) for top-down (2-stage) trackers.
    %   BgTrackWorkerObjs are tighter in terms of shapes and keeping track
    %   of numviews, numsets etc.    
    %
    % Actually, .dmcs are not used in BgTrackWorkerObj classes, so just
    % move .dmcs.
    
    % This belongs in a BgTrainWorkerObj subclass as it isn't used by
    % BgTrackWorkerObjs.
    dmcs % [nview] DeepModelChainOnDisk array  
  end
  
  methods (Abstract)
    tf = fileExists(obj,file)
    tf = errFileExistsNonZeroSize(obj,errFile)
    s = fileContents(obj,file)
    killFiles = getKillFiles(obj)
    [tf,warnings] = killProcess(obj)
    sRes = compute(obj)
  end
  
  methods
    
    function obj = BgWorkerObj(dmcs,varargin)
      if nargin == 0,
        return;
      end
      obj.dmcs = dmcs;
      obj.reset();
    end
    
    function v = n(obj)
      v = obj.dmcs.n;
    end

    function logFiles = getLogFiles(obj)
      fprintf('Using BgWorkerObj.getLogFiles ... maybe shouldn''t happen.\n');
      logFiles = {};
    end
    
    function errFile = getErrFile(obj)
      errFile = {};
    end

    function reset(obj)
      
    end

    function nframes = readTrkFileStatus(obj,f,partFileIsTextStatus)
      nframes = 0;
      if nargin < 3,
        partFileIsTextStatus = false;
      end
      if ~exist(f,'file'),
        return;
      end
      if partFileIsTextStatus,
        s = obj.fileContents(f);
        nframes = TrkFile.getNFramesTrackedPartFile(s);
      end
    end
       
    function printLogfiles(obj) % obj const
      logFiles = obj.getLogFiles();
      logFiles = unique(logFiles);
      logFileContents = cellfun(@(x)obj.fileContents(x),logFiles,'uni',0);
      BgWorkerObj.printLogfilesStc(logFiles,logFileContents)
    end

    function ss = getLogfilesContent(obj) % obj const
      logFiles = obj.getLogFiles();
      logFileContents = cellfun(@(x)obj.fileContents(x),logFiles,'uni',0);
      ss = BgWorkerObj.getLogfilesContentStc(logFiles,logFileContents);
    end
    
    function [tfEFE,errFile] = errFileExists(obj) % obj const
      errFile = obj.getErrFile();
      if isempty(errFile),
        tfEFE = false;
      else
        tfEFE = any(cellfun(@(x) obj.errFileExistsNonZeroSize(x),errFile));
      end
    end
    
    function ss = getErrorfileContent(obj) % obj const
      errFiles = obj.getErrFile();
      errFileContents = cellfun(@(x)obj.fileContents(x),errFiles,'uni',0);
      ss = BgWorkerObj.getLogfilesContentStc(errFiles,errFileContents);
      %ss = strsplit(obj.fileContents(errFile),'\n');
    end
    
    function tfLogErrLikely = logFileErrLikely(obj,file) % obj const
      tfLogErrLikely = obj.fileExists(file);
      if tfLogErrLikely
        logContents = obj.fileContents(file);
        tfLogErrLikely = ~isempty(regexpi(logContents,'exception','once'));
      end
    end

    function lsdir(obj,dir) %#ok<INUSL> 
      if ispc 
        lscmd = 'dir';
      else
        lscmd = 'ls -al';
      end
      cmd = sprintf('%s "%s"',lscmd,dir);
      system(cmd);
    end
    
    function dispProjDir(obj)
      fprintf('### %s\n',dpl);
      obj.lsdir(dpl);
      fprintf('\n');
    end
    
    function dispModelChainDir(obj)
      for i=1:obj.n,
        dmcl = dmc.dirModelChainLnx(i);
        dmcl = dmcl{1};
        [ijob,ivw,istage] = obj.dmc.ind2sub(i);
        fprintf('### Model %d, job %d, view %d, stage %d: %s\n',ivw,ijob,ivw,istage,dmcl);
        obj.lsdir(dmcl);
        fprintf('\n');
      end
    end
    
    function dispTrkOutDir(obj)
      dtol = obj.dmcs.dirTrkOutLnx;
      for i = 1:obj.n,
        [ijob,ivw,istage] = obj.dmc.ind2sub(i);
        cmd = sprintf('ls -al "%s"',dtol);
        fprintf('### Model %d, job %d, view %d, stage %d: %s\n',i,ijob,ivw,istage,dtol{i});
        system(cmd);
        fprintf('\n');
      end
    end
    
%     function backEnd = getBackEnd(obj)
%       
%       backEnd = obj.dmcs.backEnd;
%       
%     end
    
    function res = queryAllJobsStatus(obj)
      
      res = 'Not implemented.';
      
    end
    
    function res = queryMyJobsStatus(obj)
      
      res = 'Not implemented.';
      
    end
   
    function res = getIsRunning(obj)
      
      % not implemented
      res = true;
    
    end
    
  end
  
  methods (Static)
    
    function printLogfilesStc(logFiles,logFileContents)
      % Print logs for all views
      
      for ivw=1:numel(logFiles)
        logfile = logFiles{ivw};
        fprintf(1,'\n### Job %d:\n### %s\n\n',ivw,logfile);
        disp(logFileContents{ivw});
      end
    end

    function ss = getLogfilesContentStc(logFiles,logFileContents)
      % Print logs for all views

      ss = {};
      for ivw=1:numel(logFiles)
        logfile = logFiles{ivw};
        ss{end+1} = sprintf('### Job %d:',ivw); %#ok<AGROW>
        ss{end+1} = sprintf('### %s',logfile); %#ok<AGROW>
        ss{end+1} = ''; %#ok<AGROW>
        ss = [ss,strsplit(logFileContents{ivw},'\n')]; %#ok<AGROW>
      end
    end

    
  end
  
end
