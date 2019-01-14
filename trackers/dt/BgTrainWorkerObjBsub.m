classdef BgTrainWorkerObjBsub < BgTrainWorkerObjLocalFilesys
    
  methods
    
    function obj = BgTrainWorkerObjBsub(nviews,dmcs)
      obj@BgTrainWorkerObjLocalFilesys(nviews,dmcs);
    end
    
    function killJob(obj,jID)
      % jID: scalar jobID
      
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
        warningNoTrace('Bkill command failed: %s',res);
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
      
      bjobscmd = sprintf('bjobs %d; echo "More detail:"; bjobs -l %d',jID,jID);
      bjobscmd = DeepTracker.codeGenSSHGeneral(bjobscmd,'bg',false);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('Bkill command failed: %s',res);
      else
        
%         i = strfind(res,'Job <');
%         if isempty(i),
%           warning('Could not parse output from querying job status');
%           return;
%         end
%         res = res(i(1):end);
      end
      
    end
    
    function fcn = makeJobKilledPollFcn(obj,jID)
      pollcmd = sprintf('bjobs -o stat -noheader %d',jID);
      pollcmd = DeepTracker.codeGenSSHGeneral(pollcmd,'bg',false);
      
      fcn = @lcl;
      
      function tf = lcl
        % returns true when jobID is killed
        %disp(pollcmd);
        [st,res] = system(pollcmd);
        if st==0
          tf = isempty(regexp(res,'RUN','once'));
        else
          tf = false;
        end
      end
    end
    
    function tfsucc = createKillToken(obj,killtoken)
      [killdir,n] = fileparts(killtoken);
      if isempty(n),
        killdir = '.';
      end
      touchcmd = sprintf('mkdir -p %s; touch %s',killdir,killtoken);
      touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false);
      [st,res] = system(touchcmd);
      if st~=0
        tfsucc = false;
        warningNoTrace('Failed to create KILLED token: %s',killtoken);
      else
        tfsucc = true;
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',killtoken);
      end
    end
    
  end
  
end
