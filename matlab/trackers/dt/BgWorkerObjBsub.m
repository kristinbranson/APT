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
      jobid = DLBackEndClass.parseJobIDBsub(res);
%       PAT = 'Job <(?<jobid>[0-9]+)>';
%       stoks = regexp(res,PAT,'names');
%       if ~isempty(stoks)
%         jobid = str2double(stoks.jobid);
%       else
%         jobid = nan;
%         warningNoTrace('Failed to ascertain jobID.');
%       end
      fprintf('Process job (view %d, mov %d) spawned, jobid=%d.\n\n',...
        iview,imov,jobid);
      % assigning to 'local' workerobj, not the one copied to workers
      obj.jobID(imov,iview) = jobid;
      
    end
    
    function killJob(obj,jID)
      % jID: scalar jobID
      
      if isnan(jID),
        fprintf('killJob, jID is nan\n');
        return;
      end
      
      if obj.isKilled(jID),
        return;
      end
      
      bkillcmd = sprintf('bkill %d',jID);
      bkillcmd = DeepTracker.codeGenSSHGeneral(bkillcmd,'bg',false);
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
      end
    end
    
    function res = queryAllJobsStatus(obj)
      
      bjobscmd = 'bjobs';
      bjobscmd = DeepTracker.codeGenSSHGeneral(bjobscmd,'bg',false);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      else
        
%         i = strfind(res,'JOBID');
%         if isempty(i),
%           warning('Could not parse output from querying job status');
%           return;
%         end
%         res = res(i(1):end);
      end
      
    end
    
    function res = queryJobStatus(obj,jID)
      
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
      bjobscmd = DeepTracker.codeGenSSHGeneral(bjobscmd,'bg',false);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('bjobs command failed: %s',res);
      else
        
%         i = strfind(res,'Job <');
%         if isempty(i),
%           warning('Could not parse output from querying job status');
%           return;
%         end
%         res = res(i(1):end);
      end
      
    end

    function tf = isKilled(obj,jID)
      
      if isnan(jID) || isempty(jID),
        fprintf('isKilled(nan)!\n');
        tf = false;
        return;
      end
      
      runStatuses = {'PEND','RUN','PROV','WAIT'};
      
      pollcmd = sprintf('bjobs -o stat -noheader %d',jID);
      pollcmd = DeepTracker.codeGenSSHGeneral(pollcmd,'bg',false);
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
%       pollcmd = sprintf('bjobs -o stat -noheader %d',jID);
%       pollcmd = DeepTracker.codeGenSSHGeneral(pollcmd,'bg',false);
%       
%       fcn = @lcl;
%       
%       function tf = lcl
%         % returns true when jobID is killed
%         %disp(pollcmd);
%         [st,res] = system(pollcmd);
%         if st==0
%           tf = isempty(regexp(res,'RUN','once'));
%         else
%           tf = false;
%         end
%       end
    end
    
    function tfsucc = createKillToken(obj,killtoken)
      [killdir,n] = fileparts(killtoken);
      if isempty(n),
        killdir = '.';
      end
      touchcmd = sprintf('mkdir -p "%s"; touch "%s"',killdir,killtoken); % codeGenSSHGeneral uses single-quotes
      touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false); 
      [st,res] = system(touchcmd);
      if st~=0
        tfsucc = false;
        warningNoTrace('Failed to create KILLED token: %s',killtoken);
      else
        tfsucc = true;
        fprintf('Created KILLED token: %s.\nPlease wait for your monitor to acknowledge the kill!\n',killtoken);
      end
    end
    
  end
  
end
