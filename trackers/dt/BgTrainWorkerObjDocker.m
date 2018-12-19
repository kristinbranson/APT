classdef BgTrainWorkerObjDocker < BgTrainWorkerObjLocalFilesys  
  
  methods
    
    function obj = BgTrainWorkerObjDocker(nviews,dmcs)
      obj@BgTrainWorkerObjLocalFilesys(nviews,dmcs);
    end
    
    function killJob(obj,jID)
      % jID: scalar jobID
      
      bkillcmd = sprintf('docker kill %s',jID{1});
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Docker kill command failed: %s',res);
      end
    end
    
    function res = queryAllJobsStatus(obj)
      
      res = 'Not implemented';
%       bjobscmd = 'bjobs';
%       bjobscmd = DeepTracker.codeGenSSHGeneral(bjobscmd,'bg',false);
%       fprintf(1,'%s\n',bjobscmd);
%       [st,res] = system(bjobscmd);
%       if st~=0
%         warningNoTrace('Bkill command failed: %s',res);
%       end
      
    end
        
    function fcn = makeJobKilledPollFcn(obj,jID)
      jID = jID{1};
      jIDshort = jID(1:8);
      pollcmd = sprintf('docker ps -q -f "id=%s"',jIDshort);
      
      fcn = @lcl;
      
      function tf = lcl
        % returns true when jobID is killed
        %disp(pollcmd);
        [st,res] = system(pollcmd);
        if st==0
          tf = ~isempty(regexp(res,jIDshort,'once'));
        else
          tf = false;
        end
      end
    end
    
    function createKillToken(obj,killtoken)
      touchcmd = sprintf('touch %s',killtoken);
      %touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false);
      [st,res] = system(touchcmd);
      if st~=0
        warningNoTrace('Failed to create KILLED token: %s',kfile);
      else
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',kfile);
      end
    end
    
  end
  
end
