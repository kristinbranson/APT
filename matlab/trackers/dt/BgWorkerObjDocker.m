classdef BgWorkerObjDocker < BgWorkerObjLocalFilesys  
  
  % properties
  %   backend
  % end

  methods
    
    function obj = BgWorkerObjDocker(dmcs,backend,varargin)
      obj@BgWorkerObjLocalFilesys(dmcs,backend,varargin{:});
      %obj.backend = backend ;
    end
    
    function parseJobID(obj,res,iview,imov)
      % sets/initializes .jobID for given mov(job)/view(job)
      if nargin < 4,
        imov = 1;
      end
      containerID = apt.parseJobIDDocker(res);
      fprintf('Process job (movie %d, view %d) spawned, docker containerID=%s.\n\n',...
              imov,iview,containerID);
      obj.jobID{imov,iview} = containerID;
    end
    
    function killJob(obj, jobid)
      % jobid: old-style string
      if isempty(jobid),
        fprintf('killJob: jobid is empty\n');
        return
      end      
      if iscell(jobid) ,
        jobid = jobid{1};
      end
      if isempty(jobid),
        fprintf('killJob: jobid is empty\n');
        return
      end
      if obj.isKilled(jobid),
        return
      end      
      cmd = sprintf('%s kill %s', apt.dockercmd(), jobid);        
      fprintf(1,'%s\n',cmd);
      [st,res] = obj.backend.runBatchCommandOutsideContainer(cmd);  
        % It uses the docker executable, but it still runs outside the docker
        % container.
      if st~=0
        warningNoTrace('docker kill command failed: %s',res);
      end
    end  % function
    
    function res = queryAllJobsStatus(obj)      
      cmd = sprintf('%s container ls', apt.dockercmd());
      fprintf(1,'%s\n',cmd);
      [st,res] = obj.backend.runBatchCommandOutsideContainer(cmd);
      if st~=0
        warningNoTrace('docker ps command failed: %s',res);
      end
    end
    
    function res = queryJobStatus(obj,jobid)
      if isempty(jobid),
        res = 'jobid not set';
        return
      end      
      if iscell(jobid) ,
        jobid = jobid{1};
      end
      if isempty(jobid),
        res = 'jobid not set';
        return;
      end
      bjobscmd = sprintf('%s container ls -a -f id=%s',apt.dockercmd(),jobid);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = obj.backend.runBatchCommandOutsideContainer(bjobscmd);
      if st~=0
        warningNoTrace('docker ps command failed: %s',res);
      end      
    end  % function
    
    % true if the job is no longer running
    function tf = isKilled(obj,jobid)
      if isempty(jobid),
        fprintf('isKilled: jobid is empty\n');
        tf = false;
        return
      end      
      if iscell(jobid) ,
        jobid = jobid{1};
      end
      if isempty(jobid),
        fprintf('isKilled: jobid is empty\n');
        tf = false;
        return
      end
      jobidshort = jobid(1:8);
      pollcmd = sprintf('%s ps -q -f "id=%s"',apt.dockercmd(),jobidshort);
      
      [st,res] = obj.backend.runBatchCommandOutsideContainer(pollcmd);
      if st==0
        tf = isempty(regexp(res,jobidshort,'once'));
      else
        tf = false;
      end
    end
        
    function fcn = makeJobKilledPollFcn(obj,jobid)      
      fcn = @() obj.isKilled(jobid);
    end
    
    function tfsucc = createKillToken(obj,killtoken)
      p = fileparts(killtoken);
      if ~isempty(p) && ~exist(p,'dir'),
        fprintf('Directory %s does not exist, creating.\n',p);
        [tfsucc,msg] = mkdir(p);
        if ~tfsucc,
          warning('Error creating directory: %s',msg);
          return;
        end
      end
      if ispc
        touchcmd = sprintf('echo . > "%s"',killtoken);
      else
        touchcmd = sprintf('touch "%s"',killtoken);
      end
      %touchcmd = wrapCommandSSH(touchcmd);
      [st,res] = obj.backend.runBatchCommandOutsideContainer(touchcmd);
      if st~=0
        warningNoTrace('Failed to create KILLED token: %s.\n%s',killtoken,res);
        tfsucc = false;
      else
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge that the process has been killed!\n',killtoken);
        tfsucc = true;
      end
    end  % function
    
  end  % methods
  
end  % classdef
