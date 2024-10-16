classdef BgWorkerObjConda < BgWorkerObjLocalFilesys  
  % Conda jobs are just linux background processes.  So e.g. killing a job is
  % just shelling out to kill the process.  Conda backend is only supported on
  % linux, so that simplifies some things.

  methods    
    function obj = BgWorkerObjConda(varargin)
      obj@BgWorkerObjLocalFilesys(varargin{:});
    end
    
    function parseJobID(obj,res,iview,imov)
      % sets/initializes .jobID for given mov(job)/view(job)
      if nargin < 4,
        imov = 1;
      end
      jobid = apt.parseJobIDConda(res) ;
      obj.jobID{imov,iview} = jobid ;
    end
        
    function killJob(obj,jobid)  %#ok<INUSD> 
      if isempty(jobid) ,
        fprintf('killJob: jobid is empty!\n');
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end
      if isempty(jobid) ,
        fprintf('killJob: jobid is empty!\n');
        return
      end
      command_line = sprintf('kill %s', jobid) ;
      [status, stdouterr] = system(command_line) ;  % conda is Linux-only, so can just use system()
      did_kill = (status==0) ;
      if ~did_kill ,
        msg = sprintf('Unable to kill conda process with PID %s.  Command stdout/stderr was:\n%s', jobid, stdouterr) ;
        warningNoTrace(msg);
      end      
    end  % function
    
    function result = queryAllJobsStatus(obj)
      command_line = 'pgrep conda --list-full | grep --fixed-strings ''APT_interface.py'' | awk ''{print $1}''' ;      
      [status, stdouterr] = system(command_line) ;  % conda is Linux-only, so can just use system()
      if status ~= 0 ,
        result = { 'Unable to determine whether any jobs are running.' } ;
        return
      end     
      numeric_pid_from_job_index = sscanf(stdouterr, '%d')' ;  % row vector
      string_pid_from_job_index = arrayfun(@(npid)(fif(isfinite(npid), num2str(npid), '')), numeric_pid_from_job_index, 'UniformOutput', false) ;
      job_count = numel(string_pid_from_job_index) ;
      if job_count == 0 ,
        result = { 'No jobs running.' } ;
        return        
      end
      raw_result = cellfun(@(pid)(obj.queryJobStatus(pid)), string_pid_from_job_index, 'UniformOutput', false) ;
      result = raw_result(:) ;   % Want col vector of old-style strings
    end  % function
    
    function result = queryJobStatus(obj,jobid)  %#ok<INUSD> 
      % Create a string that summarizes the state of job jobid, including whether it's
      % running and when it was started.
      if isempty(jobid) ,
        result = 'jobid is empty!';
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end
      if isempty(jobid) ,
        result = 'jobid is empty!' ;
        return
      end
      command_line = sprintf('ps -p %s', jobid) ;
      [status, ~] = system(command_line) ;  % conda is Linux-only, so can just use system()
      is_running = (status==0) ;
      state_as_string = fif(is_running, 'running', 'not-running') ;
      command_line_2 = sprintf('ps -p %s -o lstart=', jobid) ;
      [status_2, stdouterr_2] = system(command_line_2) ;  % conda is Linux-only, so can just use system()
      if status_2==0 ,
        trimmed_line = strtrim(stdouterr_2) ;
        if isempty(trimmed_line) ,
          % Not sure how this would happen, but just in case
          start_time_as_string = '??' ;
        else
          start_time_as_string = trimmed_line ;
        end
      else
        start_time_as_string = '??' ;
      end
      result = sprintf('ID %s, started %s: %s', jobid, start_time_as_string, state_as_string) ;
    end  % function
    
    function tf = isKilled(obj,jobid)  %#ok<INUSD> 
      % Returns true if the job is no longer running
      if isempty(jobid) ,
        tf = false;
        fprintf('isKilled: jobid is empty!\n');
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end
      if isempty(jobid) ,
        tf = false;
        fprintf('isKilled: jobid is empty!\n');
        return
      end            
      command_line = sprintf('ps -p %s', jobid) ;
      [status, ~] = system(command_line) ;  % conda is Linux-only, so can just use system()
      is_running = (status==0) ;
      tf = ~is_running ;
    end  % function
        
    function fcn = makeJobKilledPollFcn(obj,jobid)      
      fcn = @()(obj.isKilled(jobid)) ;
    end
    
    function tfsucc = createKillToken(obj,killtoken)  %#ok<INUSD> 
      % killtoken is a string contained a filename like:
      %   /home/taylora/.apt/tp<whatever>/four_points_180806/multi_cid/view_0/20240918T220658/20240918T220659_new.KILLED
      % This function creates a zero-length file in that location.

      parent_dir = fileparts(killtoken);
      if ~isempty(parent_dir) && ~exist(parent_dir,'dir'),
        fprintf('Directory %s does not exist, creating.\n',parent_dir);
        command_line = sprintf('mkdir -p %s', escape_string_for_bash(parent_dir)) ;
        [st,res] = system(command_line) ;
        if st ~= 0 ,
          warning('Error creating directory: %s',res);
          tfsucc = false ;
          return
        end
      end
      fid = fopen(killtoken,'w');
      if fid < 0,
        warningNoTrace('Failed to create KILLED token: %s',killtoken);
        tfsucc = false;
      else
        fclose(fid);
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge that the process has been killed!\n', killtoken);
        tfsucc = true;
      end
    end  % function    
  end  % methods
end  % classdef
