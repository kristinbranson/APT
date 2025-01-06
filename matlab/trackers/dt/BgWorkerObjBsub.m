classdef BgWorkerObjBsub < BgWorkerObjLocalFilesys
    
  methods    
    function obj = BgWorkerObjBsub(varargin)
      obj@BgWorkerObjLocalFilesys(varargin{:});
    end
    
    function parseJobID(obj,res,iview,imov)
      % sets/initializes .jobID for given mov(job)/view(job)      
      if nargin < 4,
        imov = 1;
      end
      jobid = apt.parseJobIDBsub(res);
      fprintf('Process job (view %d, mov %d) spawned, jobid=%s.\n\n',...
              iview,imov,jobid);
      % assigning to 'local' workerobj, not the one copied to workers
      obj.jobID{imov,iview} = jobid;
    end
    
    function killJob(obj,jobid)
      % jobid: scalar numeric jobID, or maybe a single element cell array
      % holding a scalar numeric jobID
      if isempty(jobid) ,
        fprintf('killJob: jobid is empty!\n');
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end
      if isempty(jobid),
        fprintf('killJob, jobid is empty\n');
        return
      end
      if obj.isKilled(jobid),
        return
      end
      bkillcmd = sprintf('bkill %s',jobid);
      bkillcmd = wrapCommandSSH(bkillcmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
      end
    end
    
    function res = queryAllJobsStatus(obj)  %#ok<MANU> 
      bjobscmd = 'bjobs';
      bjobscmd = wrapCommandSSH(bjobscmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      else
      end
    end
    
    function res = queryJobStatus(obj,jobid)
      if isempty(jobid) ,
        res = sprintf('jobid is empty!\n');
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end
      try
        tfKilled = obj.isKilled(jobid);
        if tfKilled,
          res = sprintf('Job %s has been killed',jobid);
          return;
        end
      catch
        fprintf('Failed to poll for job %s before killing\n',jobid);
      end
      bjobscmd = sprintf('bjobs %s; echo "More detail:"; bjobs -l %s',jobid,jobid);
      bjobscmd = wrapCommandSSH(bjobscmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      end      
    end

    function tf = isKilled(obj,jobid)  %#ok<INUSD> 
      if isempty(jobid) ,
        fprintf('isKilled: jobid is empty!\n');
        tf = false;
        return
      end
      if iscell(jobid)  ,
        jobid = jobid{1};
      end        
      if isempty(jobid) ,
        fprintf('isKilled: jobid is empty!\n');
        tf = false;
        return
      end
      runStatuses = {'PEND','RUN','PROV','WAIT'};
      pollcmd = sprintf('bjobs -o stat -noheader %s',jobid);
      pollcmd = wrapCommandSSH(pollcmd,'host',DLBackEndClass.jrchost);
      [st,res] = system(pollcmd);
      if st==0
        s = sprintf('(%s)|',runStatuses{:});
        s = s(1:end-1);
        tf = isempty(regexp(res,s,'once'));
      else
        tf = false;
      end
    end
    
    function fcn = makeJobKilledPollFcn(obj,jobid)      
      fcn = @() obj.isKilled(jobid);
    end
    
    function tfsucc = createKillToken(obj,killtoken)  %#ok<INUSD> 
      [killdir,n] = fileparts(killtoken);
      if isempty(n),
        killdir = '.';
      end
      touchcmd = sprintf('mkdir -p "%s"; touch "%s"',killdir,killtoken); % wrapCommandSSH uses single-quotes
      touchcmd = wrapCommandSSH(touchcmd,'host',DLBackEndClass.jrchost); 
      [st,res] = system(touchcmd);
      if st~=0
        tfsucc = false;
        warningNoTrace('Failed to create KILLED token: %s\nReason:\n%s',killtoken,res);
      else
        tfsucc = true;
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge that the process has been killed!\n',killtoken);
      end
    end  % function    
  end  % methods
end  % classdef
