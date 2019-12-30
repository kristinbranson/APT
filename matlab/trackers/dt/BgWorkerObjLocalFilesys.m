classdef BgWorkerObjLocalFilesys < BgWorkerObj
  %
  % 
  % 1. Artifacts written to local filesys
  % 2. Process killed by sending message, polling to confirm, and touching
  % filesystem tok
  %
  
  properties
    jobID % [nmov x nviewJobs] bsub jobID; or docker cellstr containerID
    
    killPollIterWaitTime = 1; % sec
    killPollMaxWaitTime = 12; % sec
  end
  
  methods (Abstract)
    killJob(obj,jID) % kill a single job. jID is scalar jobID
    fcn = makeJobKilledPollFcn(obj,jID) % create function that returns true when job is confirmed killed. jID is scalar jobID
    createKillToken(obj,killtoken) % create/touch filesystem KILL token. killtoken is full linux path
    %queryClusterJobs(obj)
  end
  
  methods
    
    function obj = BgWorkerObjLocalFilesys(varargin)
      obj@BgWorkerObj(varargin{:});
    end
    
    function tf = fileExists(~,file)
      tf = exist(file,'file')>0;
    end
    
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgWorkerObjLocalFilesys.errFileExistsNonZeroSizeStc(errFile);
    end
        
    function s = fileContents(~,file)
      if exist(file,'file')==0
        s = '<file does not exist>';
      else
        lines = readtxtfile(file);
        s = sprintf('%s\n',lines{:});
      end
    end
    
    function [tfsucc,warnings] = killProcess(obj)
      tfsucc = false;
      warnings = {};
      
      killfiles = obj.getKillFiles();
      killfiles = unique(killfiles);
      jobids = obj.jobID;
      assert(isequal(numel(jobids),numel(killfiles)));
      
      for ivw=1:numel(jobids),
        obj.killJob(jobids(ivw));
      end

      iterWaitTime = obj.killPollIterWaitTime;
      maxWaitTime = obj.killPollMaxWaitTime;

      for ivw=1:numel(jobids),
        fcn = obj.makeJobKilledPollFcn(jobids(ivw));
        tfsucc = waitforPoll(fcn,iterWaitTime,maxWaitTime);
        
        if ~tfsucc
          warnings{end+1} = 'Could not confirm that job was killed.'; %#ok<AGROW>
          warningNoTrace('Could not confirm that job was killed.');
        else
          % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to
          % pick up
          
          kfile = killfiles{ivw};
          tfsucc = obj.createKillToken(kfile);
          if ~tfsucc,
            warnings{end+1} = sprintf('Failed to create KILLED token: %s',kfile); %#ok<AGROW>
          end
        end
        
        % bgTrnMonitor should pick up KILL tokens and stop bg trn monitoring
      end
    end
    
    function res = queryMyJobsStatus(obj)
      
      jobids = obj.jobID;
      nvw = obj.nviews;
      %assert(isequal(nvw,numel(jobids)));
      
      res = repmat({''},[1,nvw]);
      for ivw=1:numel(jobids),
        res{ivw} = obj.queryJobStatus(jobids(ivw));
      end
      
    end
    
    function tfIsRunning = getIsRunning(obj)
      ids = obj.jobID;
      tfIsRunning = true(size(ids));
      try
        for i = 1:numel(ids),
          if iscell(ids),
            id = ids{i};
            if isempty(id),
              continue;
            end
          else
            id = ids(i);
            if isnan(id),
              continue;
            end
          end
          % Guard against nan jobIDs for now. This can occur when a job
          % is started; the monitor is started before the job is spawned,
          % and the monitorviz calls this after receiving the first
          % result from the bg worker before the jID is set.
          tfIsRunning(i) = ~obj.isKilled(id);
        end
        %fprintf('isRunning = %s\n',mat2str(tfIsRunning));
      catch ME,
        fprintf('Error in updateIsRunning:\n%s\n',getReport(ME));
      end
    end
        
  end
    
  methods (Static)
    function tfErrFileErr = errFileExistsNonZeroSizeStc(errFile)
      tfErrFileErr = exist(errFile,'file')>0;
      if tfErrFileErr
        direrrfile = dir(errFile);
        tfErrFileErr = direrrfile.bytes>0;
      end
    end
  end
  
end
