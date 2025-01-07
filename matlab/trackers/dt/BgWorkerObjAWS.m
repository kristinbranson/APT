classdef BgWorkerObjAWS < BgWorkerObj
  
  properties
    jobID  % [nmovjob x nviewJobs] remote PIDs
    backend  % Instance of DLBackEndClass, protected "by convention"
  end
  
  methods (Access=protected)    
    function obj2 = copyElement(obj)
      % overload so that .backend is deep-copied
      obj2 = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.dmcs)
        obj2.dmcs = copy(obj.dmcs);
      end
      if ~isempty(obj.backend)
        obj2.backend = copy(obj.backend);
      end
    end
  end  % protected methods block

  methods
    function obj = BgWorkerObjAWS(dmcs,backend,varargin)
      obj@BgWorkerObj(dmcs);
      obj.backend = backend;
    end
    
    function obj2 = copyAndDetach(obj)
      % See note in BgClient/configure(). We create a new obj2 here that is
      % a deep-copy made palatable for parfeval      
      obj2 = copy(obj); % deep-copies obj, including .backend and .dmcs if appropriate
    end    
    
    function tf = fileExists(obj, f)
      tf = obj.backend.fileExists(f);
    end
    
    function tf = fileExistsAndIsNonempty(obj,errFile)
      tf = obj.backend.fileExistsAndIsNonempty(errFile);
    end    
    
    function s = fileContents(obj,f)
      s = obj.backend.fileContents(f);
    end

    function tfsucc = lsdir(obj,dir)
      tfsucc = obj.backend.lsdir(dir);
    end
    
    function result = fileModTime(obj, filename)
      result = obj.backend.fileModTime(filename) ;
    end

%     function [tfsucc,warnings] = killProcess(obj)
%       warnings = {};
% %       if ~obj.isRunning
% %         error('Training is not in progress.');
% %       end
%       backend = obj.backend ;
%       if isempty(backend)
%         error('backend object is unset.');
%       end
% 
%       % if ~ec2.canKillRemoteProcess()
%       %   tfpid = ec2.getRemotePythonPID();
%       %   if ~tfpid
%       %     error('Could not ascertain remote process ID in AWSEC2 instance %s.',ec2.instanceID);
%       %   end
%       % end
% 
%       killfile = obj.getKillFiles();
%       killfile = unique(killfile);
%       assert(isscalar(killfile)); % for now
%       killfile = killfile{1};
% 
%       backend.killRemoteProcess();
% 
%       % expect command to fail; fail -> py proc killed
%       %pollCbk = @()~aws.runBatchCommandOutsideContainer('pgrep -o python','dispcmd',true,'failbehavior','silent');
%       pollCbk = @()backend.getNoPyProcRunning();
%       iterWaitTime = 1;
%       maxWaitTime = 20;
%       tfsucc = waitforPoll(pollCbk,iterWaitTime,maxWaitTime);
% 
%       if ~tfsucc
%         warningNoTrace('Could not confirm that remote process was killed.');
%         warnings{end+1} = 'Could not confirm that remote process was killed.';
%         return
%       end
%       % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to 
%       % pick up
%       killfile_folder_path = fileparts(killfile) ;
%       escaped_killfile_folder_path = escape_string_for_bash(killfile_folder_path) ;
%       cmd = sprintf('mkdir -p %s',escaped_killfile_folder_path); 
%       st = backend.runBatchCommandOutsideContainer(cmd);
%       tfsucc = (st==0) ;
%       if ~tfsucc ,
%         warningNoTrace('Failed to create remote KILLED token dir: %s',killfile_folder_path);
%         warnings{end+1} = sprintf('Failed to create remote KILLED token dir: %s',killfile_folder_path);          
%         return
%       end
% 
%       escaped_killfile = escape_string_for_bash(killfile) ;
%       cmd = sprintf('touch %s',escaped_killfile);
%       st = backend.runBatchCommandOutsideContainer(cmd);
%       tfsucc = (st==0) ;
%       if ~tfsucc
%         warningNoTrace('Failed to create remote KILLED token: %s',killfile);
%         warnings{end+1} = sprintf('Failed to create remote KILLED token: %s',killfile);
%         return
%       end
%       fprintf('Created remote KILLED token: %s. Please wait for your training monitor to acknowledge that the process has been killed!\n',killfile);
%       % bgTrnMonitorAWS should pick up KILL tokens and stop bg trn monitoring
%     end  % function
    
  end  % methods
    
end  % classdef
