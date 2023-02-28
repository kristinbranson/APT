classdef BgWorkerObjConda < BgWorkerObjLocalFilesys  
  
  methods
    
    function obj = BgWorkerObjConda(varargin)
      obj@BgWorkerObjLocalFilesys(varargin{:});
    end
    
    function parseJobID(obj,res,iview,imov)
      % sets/initializes .jobID for given mov(job)/view(job)

      if nargin < 4,
        imov = 1;
      end
      obj.jobID{imov,iview} = res;
      
    end
        
    function killJob(obj,jID)
      % jID: scalar FevalFuture jobID 
      
      if iscell(jID) && ~isempty(jID),
        jID = jID{1};
      end
      if isempty(jID),
        fprintf('killJob: jID is empty\n');
        return;
      end
      jID.cancel();
      if ~strcmp(jID.State,'finished'),
        msg = sprintf('%s\n',jID.Error.message);
        warningNoTrace('Canceling job failed: %s',msg);
      end
    end
    
    function res = queryAllJobsStatus(obj)
    
      p = gcp;
      q = p.FevalQueue;
      r = q.RunningFutures;
      res = {};
      if isempty(r),
        res{end+1} = 'No jobs running.';
      else
        res{end+1} = 'Jobs running:';
        for i = 1:numel(r),
          res{end+1} = sprintf('ID %d, started %s: %s',r(i).ID,r(i).StartDateTime,r(i).State);
        end
      end
      r = q.QueuedFutures;
      if isempty(r),
        res{end+1} = 'No jobs queued.';
      else
        res{end+1} = 'Jobs queued:';
        for i = 1:numel(r),
          res{end+1} = sprintf('ID %d, started %s: %s',r(i).ID,r(i).StartDateTime,r(i).State);
        end
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
      res = sprintf('ID %d, started %s: %s',jID.ID,jID.StartDateTime,jID.State);
      
    end
    
    % true if the job is no longer running
    function tf = isKilled(obj,jID)
      
      if iscell(jID),
        jID = jID{1};
      end
      if isempty(jID),
        tf = false;
        fprintf('isKilled: jID is empty!\n');
        return;
      end
      tf = strcmp(jID.State,'finished');
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
      fid = fopen(killtoken,'w');
      if fid < 0,
        warningNoTrace('Failed to create KILLED token: %s.\n%s',killtoken,res);
        tfsucc = false;
      else
        fclose(fid);
        fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',killtoken);
        tfsucc = true;
      end
    end
    
  end
  
end
