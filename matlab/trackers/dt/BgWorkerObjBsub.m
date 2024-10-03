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
      fprintf('Process job (view %d, mov %d) spawned, jobid=%d.\n\n',...
        iview,imov,jobid);
      % assigning to 'local' workerobj, not the one copied to workers
      obj.jobID(imov,iview) = jobid;
    end
    
    function killJob(obj,jID)
      % jID: scalar numeric jobID, or maybe a single element cell array
      % holding a scalar numeric jobID
      if isempty(jID) ,
        fprintf('killJob: jID is empty!\n');
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end
      if isnan(jID),
        fprintf('killJob, jID is nan\n');
        return
      end
      if obj.isKilled(jID),
        return
      end
      bkillcmd = sprintf('bkill %d',jID);
      bkillcmd = wrapCommandSSH(bkillcmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
      end
    end
    
    function res = queryAllJobsStatus(obj)      
      bjobscmd = 'bjobs';
      bjobscmd = wrapCommandSSH(bjobscmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      else
      end
    end
    
    function res = queryJobStatus(obj,jID)
      if isempty(jID) ,
        res = sprintf('jID is empty!\n');
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end
      try
        tfKilled = obj.isKilled(jID);
        if tfKilled,
          res = sprintf('Job %d has been killed',jID);
          return;
        end
      catch
        fprintf('Failed to poll for job %d before killing\n',jID);
      end
      bjobscmd = sprintf('bjobs %d; echo "More detail:"; bjobs -l %d',jID,jID);
      bjobscmd = wrapCommandSSH(bjobscmd,'host',DLBackEndClass.jrchost);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      end      
    end

    function tf = isKilled(obj,jID) 
      if isempty(jID) ,
        fprintf('isKilled: jID is empty!\n');
        tf = false;
        return
      end
      if iscell(jID)  ,
        jID = jID{1};
      end        
      if isempty(jID) || isnan(jID) ,
        fprintf('isKilled: jID is nan!\n');
        tf = false;
        return
      end
      runStatuses = {'PEND','RUN','PROV','WAIT'};
      pollcmd = sprintf('bjobs -o stat -noheader %d',jID);
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
    
    function fcn = makeJobKilledPollFcn(obj,jID)      
      fcn = @() obj.isKilled(jID);
    end
    
    function tfsucc = createKillToken(obj,killtoken)
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
