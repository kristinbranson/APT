classdef BgWorkerObjDocker < BgWorkerObjLocalFilesys  
  
  methods
    
    function obj = BgWorkerObjDocker(varargin)
      obj@BgWorkerObjLocalFilesys(varargin{:});
    end
    
    function parseJobID(obj,res,iview,imov)
      
      if nargin < 4,
        imov = 1;
      end
      containerID = BgWorkerObjDocker.parseJobIDStatic(res);
      fprintf('Process job (movie %d, view %d) spawned, docker containerID=%s.\n\n',...
        imov,iview,containerID);
      
%       containerID = strtrim(res);
%       fprintf('Process job (view %d) spawned, docker containerID=%s.\n\n',...
%         iview,containerID);
      % assigning to 'local' workerobj, not the one copied to workers
      obj.jobID{imov,iview} = containerID;
    end
    
    function s = dockercmd(obj)
      
      if isempty(APT.DOCKER_REMOTE_HOST),
        s = 'docker';
      else
        s = sprintf('ssh -t %s docker',APT.DOCKER_REMOTE_HOST);
      end

    end
    
    function killJob(obj,jID)
      % jID: scalar jobID

      if isempty(jID) || isempty(jID{1}),
        fprintf('killJob: jID is empty\n');
        return;
      end

      if obj.isKilled(jID),
        return;
      end
      
      bkillcmd = sprintf('%s kill %s',obj.dockercmd,jID{1});
        
      fprintf(1,'%s\n',bkillcmd);
      [st,res] = system(bkillcmd);
      if st~=0
        warningNoTrace('Docker kill command failed: %s',res);
      end
    end
    
    function res = queryAllJobsStatus(obj)
      
      bjobscmd = sprintf('%s container ls',obj.dockercmd);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('docker ps command failed: %s',res);
      else
        
%         i = strfind(res,'Job <');
%         if isempty(i),
%           warning('Could not parse output from querying job status');
%           return;
%         end
%         res = res(i(1):end);
      end
      
      
    end
    
    function res = queryJobStatus(obj,jID)
      
      if iscell(jID),
        jID = jID{1};
      end
      if isempty(jID),
        res = 'jID not set';
        return;
      end
      bjobscmd = sprintf('%s container ls -a -f id=%s',obj.dockercmd,jID);
      fprintf(1,'%s\n',bjobscmd);
      [st,res] = system(bjobscmd);
      if st~=0
        warningNoTrace('docker ps command failed: %s',res);
      else
        
%         i = strfind(res,'Job <');
%         if isempty(i),
%           warning('Could not parse output from querying job status');
%           return;
%         end
%         res = res(i(1):end);
      end
      
    end
    
    % true if the job is no longer running
    function tf = isKilled(obj,jID)
      
      jID = jID{1};
      if isempty(jID),
        tf = false;
        fprintf('isKilled: jID is empty!\n');
        return;
      end
      jIDshort = jID(1:8);
      pollcmd = sprintf('%s ps -q -f "id=%s"',obj.dockercmd,jIDshort);
      
      [st,res] = system(pollcmd);
      if st==0
        tf = isempty(regexp(res,jIDshort,'once'));
      else
        tf = false;
      end
    end
        
    function fcn = makeJobKilledPollFcn(obj,jID)
      
      fcn = @() obj.isKilled(jID);
      
%       jID = jID{1};
%       jIDshort = jID(1:8);
%       pollcmd = sprintf('%s ps -q -f "id=%s"',obj.dockercmd,jIDshort);
%       
%       fcn = @lcl;
%       
%       function tf = lcl
%         % returns true when jobID is killed
%         %disp(pollcmd);
%         [st,res] = system(pollcmd);
%         if st==0
%           tf = isempty(regexp(res,jIDshort,'once'));
%         else
%           tf = false;
%         end
%       end
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
      touchcmd = sprintf('touch %s',killtoken);
      %touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false);
      [st,res] = system(touchcmd);
      if st~=0
        warningNoTrace('Failed to create KILLED token: %s.\n%s',killtoken,res);
        tfsucc = false;
      else
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',killtoken);
        tfsucc = true;
      end
    end
    
  end
  
  methods (Static)
    
    function containerID = parseJobIDStatic(res)
      
      res = regexp(res,'\n','split');
      res = regexp(res,'^[0-9a-f]+$','once','match');
      l = cellfun(@numel,res);
      res = res{find(l==64,1)};
      assert(~isempty(res));
      containerID = strtrim(res);
      
    end
    
  end
  
end
