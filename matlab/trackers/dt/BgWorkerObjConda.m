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
      jID = apt.parseJobIDConda(res) ;
      obj.jobID{imov,iview} = jID ;
    end
        
    function killJob(obj,jID)  %#ok<INUSD> 
      if isempty(jID) ,
        fprintf('killJob: jID is empty!\n');
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end
      if isnan(jID) ,
        fprintf('killJob: jID is nan!\n');
        return
      end
      command_line = sprintf('kill %d', jID) ;
      [status, stdouterr] = system(command_line) ;  % conda is Linux-only, so can just use system()
      did_kill = (status==0) ;
      if ~did_kill ,
        msg = sprintf('Unable to kill conda process with PID %d.  Command stdout/stderr was:\n%s', jID, stdouterr) ;
        warningNoTrace(msg);
      end      
    end  % function
    
    function res = queryAllJobsStatus(obj)  %#ok<MANU> 
      % TODOALT: Need to implement this properly
      warning('Need to implement BgWorkerObjConda::queryAllJobsStatus() properly') ;
      res = {'<Not implemented>'} ;
%       p = gcp();
%       q = p.FevalQueue;
%       rfs = q.RunningFutures;
%       res = {};
%       if isempty(rfs),
%         res{end+1} = 'No jobs running.';
%       else
%         res{end+1} = 'Jobs running:';
%         for i = 1:numel(rfs),
%           res{end+1} = sprintf('ID %d, started %s: %s',rfs(i).ID,rfs(i).StartDateTime,rfs(i).State); %#ok<AGROW> 
%         end
%       end
%       qfs = q.QueuedFutures;
%       if isempty(qfs),
%         res{end+1} = 'No jobs queued.';
%       else
%         res{end+1} = 'Jobs queued:';
%         for i = 1:numel(qfs),
%           res{end+1} = sprintf('ID %d, started %s: %s',qfs(i).ID,qfs(i).StartDateTime,qfs(i).State); %#ok<AGROW> 
%         end
%       end
    end  % function
    
    function result = queryJobStatus(obj,jID)  %#ok<INUSD> 
      % Create a string that summarizes the state of job jID, including whether it's
      % running and when it was started.
      if isempty(jID) ,
        result = 'jID is empty!';
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end
      if isnan(jID) ,
        result = 'jID is nan!' ;
        return
      end      
      command_line = sprintf('ps -p %d', jID) ;
      [status, ~] = system(command_line) ;  % conda is Linux-only, so can just use system()
      is_running = (status==0) ;
      state_as_string = fif(is_running, 'running', 'not-running') ;
      command_line_2 = sprintf('ps -p %d -o lstart=', jID) ;
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
      result = sprintf('ID %d, started %s: %s', jID, start_time_as_string, state_as_string) ;
    end  % function
    
    function tf = isKilled(obj,jID)  %#ok<INUSD> 
      % Returns true if the job is no longer running
      if isempty(jID) ,
        tf = false;
        fprintf('isKilled: jID is empty!\n');
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end
      if isnan(jID) ,
        tf = false;
        fprintf('isKilled: jID is nan!\n');
        return
      end            
      command_line = sprintf('ps -p %d', jID) ;
      [status, ~] = system(command_line) ;  % conda is Linux-only, so can just use system()
      is_running = (status==0) ;
      tf = ~is_running ;
    end  % function
        
    function fcn = makeJobKilledPollFcn(obj,jID)      
      fcn = @()(obj.isKilled(jID)) ;
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
