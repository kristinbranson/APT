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
    % For BgTrainWorkerObjs, nviews is now guaranteed to equal numel(dmcs).
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
    nviews
    
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
    
    function obj = BgWorkerObj(nviews,dmcs,varargin)
      if nargin == 0,
        return;
      end
      obj.nviews = nviews;
      assert(numel(dmcs)==nviews);
      obj.dmcs = dmcs;
      obj.reset();
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
    
    function dispProjDir(obj)
      if ispc 
        lscmd = 'dir';
      else
        lscmd = 'ls -al';
      end
      ds = {obj.dmcs.dirProjLnx}';
      ds = unique(ds);
      for i=1:numel(ds)
        cmd = sprintf('%s "%s"',lscmd,ds{i});
        fprintf('### %s\n',ds{i});
        system(cmd);
        fprintf('\n');
      end
    end
    
    function dispModelChainDir(obj)
      if ispc 
        lscmd = 'dir';
      else
        lscmd = 'ls -al';
      end
      for ivw=1:obj.nviews
        dmc = obj.dmcs(ivw);
        cmd = sprintf('%s "%s"',lscmd,dmc.dirModelChainLnx);
        fprintf('### View %d: %s\n',ivw,dmc.dirModelChainLnx);
        system(cmd);
        fprintf('\n');
      end
    end
    
    function dispTrkOutDir(obj)
      for ivw=1:obj.nviews
        dmc = obj.dmcs(ivw);
        if isnan(dmc)
          % AL: This is prob not quite right in all cases (multiview,
          % MATD/BU, etc.) but this is a dev/debug method.
          dirTrk = fileparts(obj.artfctTrkfiles{1,ivw,1});
        else
          dirTrk = dmc.dirTrkOutLnx;
        end
        if ispc
          cmd = 'dir';
        else
          cmd = 'ls -al';
        end
        cmd = sprintf('%s "%s"',cmd,dirTrk);
        fprintf('### View %d: %s\n',ivw,dirTrk);
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