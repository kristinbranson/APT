classdef AWSec2 < handle
  % Object to handle specific aspects of the AWS backend.
  % 
  % This is copyable with the default copyElement() methods.  the only arguably
  % sensitive thing is the instanceIP, which is only valid for a single run of
  % the instance.  That property is Copyable but Transient.  Since Transient
  % props don't survive getting serialized and passed to parfeval, we take steps
  % to restore it on the other side of the parfeval call.  See the methods
  % packParfevalSuitcase() and restoreAfterParfeval().

  properties (Constant)
    AWS_SECURITY_GROUP = 'apt_dl';      % name of AWS security group
    % AMI = 'ami-0168f57fb900185e1';  TF 1.6
    % AMI = 'ami-094a08ff1202856d6'; TF 1.13
    % AMI = 'ami-06863f1dcc6923eb2'; % Tf 1.15 py3
    %AMI = 'ami-061ef1fe3348194d4'; % TF 1.15 py3 and python points to python3
    AMI = 'ami-09b1db2d5c1d91c38';  
      % This is the "Deep Learning Base Proprietary Nvidia Driver GPU AMI (Ubuntu
      % 20.04) 20240101" provided by Amazon. Note that this is completely generic.
      % It's not customized in any way for APT.  We do that ourselves when we create
      % an EC2 instance from this AMI.
    autoShutdownAlarmNamePat = 'aptAutoShutdown'; 
    remoteHomeDir = '/home/ubuntu'
    remoteDLCacheDir = linux_fullfile(AWSec2.remoteHomeDir, 'cacheDL')
    remoteMovieCacheDir = linux_fullfile(AWSec2.remoteHomeDir, 'movies')
    remoteAPTSourceRootDir = linux_fullfile(AWSec2.remoteHomeDir, 'APT')
    remoteTorchHomeDir = linux_fullfile(AWSec2.remoteHomeDir, 'torch')
    % scpCmd = AWSec2.computeScpCmd()
    rsyncCmd = AWSec2.computeRsyncCmd()
    instanceType = 'p3.2xlarge'  % the AWS EC2 machine instance type to use when creating a new instance
  end
  
  properties
    instanceID_ = ''  % Durable identifier for the AWS EC2 instance.  E.g.'i-07a3a8281784d4a38'.
  end

  properties (Dependent)
    instanceID
    isInstanceIDSet    % Whether the instanceID is set or not
    areCredentialsSet  % Whether the security credentials for the instance are set
    isInDebugMode      % Whether the object is in debug mode.  See isInDebugMode_
  end

  properties
    keyName = ''  % key(pair) name used to authenticate to AWS EC2, e.g. 'alt_taylora-ws4'
    pem = ''  % path to .pem file that holds an RSA private key used to ssh into the AWS EC2 instance
  end

  properties (Transient, SetAccess=protected)
    % The backend keeps track of whether the project cache is local or remote.  When it's
    % remote, we substitute the remote project cache path for the local one wherever it
    % appears.
    isProjectCacheRemote_ = false
      % True iff the "current" version of the project cache is on a remote AWS filesystem.  
      % Underscore means "protected by convention"    
    wslProjectCachePath_ = '' ;  % e.g. /groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5
    % wslTorchCachePath_ = '' ;  % e.g. /groups/branson/home/bransonk/.apt/torch
    %remoteDMCRootDir_ = '' ;  % e.g. /home/ubuntu/cacheDL    

    % Used to keep track of whether movies have been uploaded or not.
    % Transient and protected in spirit.
    didUploadMovies_ = false

    % When we upload movies, keep track of the correspondence, so we can help the
    % consumer map between the paths.  Transient, protected in spirit.
    wslPathFromMovieIndex_ = cell(1,0) ;
    remotePathFromMovieIndex_ = cell(1,0) ;
  end

  properties (Dependent)
    isProjectCacheRemote
    isProjectCacheLocal    
    wslProjectCachePath
    % wslTorchCachePath
    %remoteDMCRootDir
  end

  % Transient properties don't get copied over when you pass an AWSec2 in an arg to parfeval()!
  % So for some properties we take steps to make sure they get restored in the
  % background process, because we want them to survive the transit through the parfeval boundary.
  % (See the methods packParfevalSuitcase() and restoreAfterParfeval().)

  properties (Transient)  
    %remotePID = ''  % The PID of the Python process on the EC2 instance.  An old-style string
    isInDebugMode_ = false  % In debug mode or not.  In debug mode, for instance, AWS alarms are turned off.
    instanceIP = '' % The IP address of the EC2 instance.  An old-style string.
  end
  
  properties (Transient, NonCopyable)
    wasInstanceStarted_ = false  % This is true iff obj started the AWS EC2 instance.  If something/someone other than 
                                 % *this object* started the instance, this is false. 
                                 %
                                 % We don't want this to be copied over when passing an AWSec2 in an arg to
                                 % parfeval(), or when the object is persisted, so we make it transient.
  end

  methods    
    function obj = AWSec2()
    end
    
    function delete(obj)  %#ok<INUSD> 
      % NOTE: for now, lifecycle of obj is not tied at all to the actual
      % instance-in-the-cloud
      %fprintf('Deleting an AWSec2 object.\n') ;
    end    
  end

  methods    
    function v = get.areCredentialsSet(obj)
      v = ~isempty(obj.pem) && ~isempty(obj.keyName);
    end

    function v = get.isInstanceIDSet(obj)
      v = ~isempty(obj.instanceID_);
    end

    function result = get.isInDebugMode(obj)
      result = obj.isInDebugMode_ ;
    end

    function set.isInDebugMode(obj, value)
      obj.isInDebugMode_ = value ;
    end

    function result = get.instanceID(obj)
      result = obj.instanceID_ ;
    end

    function set.instanceID(obj, instanceID)
      if ~isempty(obj.instanceID_),
        if strcmp(instanceID, obj.instanceID_)
          return
        end
        [tfexist,tfrunning] = obj.inspectInstance();
        if tfexist && tfrunning,
          tfsucc = obj.stopInstance();
          if ~tfsucc,
            warning('Error stopping old AWS EC2 instance %s.',obj.instanceID_);
          end
        end
      end
      obj.instanceID_ = instanceID ;
      if obj.isInstanceIDSet ,
        obj.configureAlarm();
      end
    end

    function [tfsucc, instanceID] = launchNewInstance(obj, varargin)
      % Launch a brand-new instance.
      obj.instanceID = '' ;  % This calls a setter function, which stops the original instance, if any.
      cmd = AWSec2.launchInstanceCmd(obj.keyName,'instType',AWSec2.instanceType);
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (st==0) ;
      if ~tfsucc
        instanceID = '' ;
        return
      end
      json = jsondecode(json);
      instanceID = json.Instances.InstanceId;
      obj.instanceID = instanceID ;  % This calls a setter function, which e.g. configures the alarms on the new instance
      tfsucc = obj.waitForInstanceStart();
    end
    
    function [tfexist,tfrunning,json] = inspectInstance(obj,varargin)
      % Check that a specified instance exists; check if it is running; 
      % get json; sets .instanceIP if running
      % 
      % * tfexist is returned as true of the instance exists in some state.
      % * tfrunning is returned as true if the instance exists and is running.
      % * json is valid only if tfexist==true.
      
      assert(obj.isInstanceIDSet,'AWS EC2 instance ID is not set.');
      
      % Aside: this works with empty .instanceID_ if there is only one 
      % instance in the cloud, but we are not interested in that for now
      cmd = AWSec2.describeInstancesCmd(obj.instanceID_); 
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfexist = (st==0) ;
      if ~tfexist
        tfrunning = false;
        return;
      end
      json = jsondecode(json);
      if isempty(json.Reservations),
        tfexist = false;
        tfrunning = false;
        return;
      end
      
      inst = json.Reservations.Instances;
      assert(strcmp(obj.instanceID_,inst.InstanceId));
      if strcmpi(inst.State.Name,'terminated'),
        tfexist = false;
        tfrunning = false;
        return;
      end
      
      tfrunning = strcmp(inst.State.Name,'running');
      if tfrunning
        obj.instanceIP = inst.PublicIpAddress;
        fprintf('EC2 instanceID %s is running with IP %s.\n',...
          obj.instanceID_,obj.instanceIP);
      else
        % leave IP for now even though may be outdated
      end
    end  % function
    
    function [tfsucc,state,json] = getInstanceState(obj)
      assert(obj.isInstanceIDSet);
      state = '';
      cmd = AWSec2.describeInstancesCmd(obj.instanceID_); % works with empty .instanceID if there is only one instance
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (st==0) ;
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      state = json.Reservations.Instances.State.Name;      
    end
    
    function [tfsucc,json] = stopInstance(obj)
      if ~obj.isInstanceIDSet || ~obj.wasInstanceStarted_ ,
        tfsucc = true;
        json = {};
        return
      end
      fprintf('Stopping AWS EC2 instance %s...\n',obj.instanceID_);
      cmd = AWSec2.stopInstanceCmd(obj.instanceID_);
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (st==0) ;
      %obj.ClearStatus();
      if ~tfsucc
        return
      end
      json = jsondecode(json);
      obj.stopAlarm();
    end  % function    

    function [tfsucc,instanceIDs,instanceTypes,json] = listInstances(obj)    
      instanceIDs = {};
      instanceTypes = {};
      cmd = AWSec2.listInstancesCmd(obj.keyName,'instType',[]); % empty instType to list all instanceTypes
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (st==0) ;
      if tfsucc,
        info = jsondecode(json);
        if ~isempty(info.Reservations),
          instanceIDs = arrayfun(@(x)x.Instances.InstanceId,info.Reservations,'uni',0);
          instanceTypes = arrayfun(@(x)x.Instances.InstanceType,info.Reservations,'uni',0);
        end
      end
    end

    function [tfsucc,json,warningstr,state] = startInstance(obj,varargin)
      %obj.SetStatus(sprintf('Starting instance %s',obj.instanceID_));
      [doblock] = myparse(varargin,'doblock',true);
      
      %maxwaittime = 100;
      %iterwaittime = 5;
      warningstr = '';
      [tfsucc,state,json] = obj.getInstanceState();
      if ~tfsucc,
        warningstr = 'Failed to get instance state.';
        %obj.ClearStatus();
        return;
      end

      if ismember(lower(state),{'shutting-down','terminated'}),
        warningstr = sprintf('Instance is %s, cannot start',state);
        tfsucc = false;
        %obj.ClearStatus();
        return
      end
      if ismember(lower(state),{'stopping'}),
        warningstr = sprintf('Instance is %s, please wait for this to finish before starting.',state);
        tfsucc = false;
        %obj.ClearStatus();
        return;
      end
      if ~ismember(lower(state),{'running','pending'}),
        cmd = AWSec2.startInstanceCmd(obj.instanceID_);
        %[tfsucc,json] = AWSec2.syscmd(cmd,'isjsonout',true);
        [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
        tfsucc = (st==0) ;        
      end
      if ~tfsucc
        %obj.ClearStatus();
        return
      end
      json = jsondecode(json);
      if ~doblock,
        %obj.ClearStatus();
        return
      end
      
      [tfsucc] = obj.waitForInstanceStart();
      if ~tfsucc,
        warningstr = 'Timed out waiting for AWS EC2 instance to spool up.';
        %obj.ClearStatus();
        return
      end
      
      obj.inspectInstance();
      obj.configureAlarm();
      %obj.ClearStatus();
      obj.wasInstanceStarted_ = true ;
    end  % function
    
    function tfsucc = waitForInstanceStart(obj)
      maxwaittime = 100;
      iterwaittime = 5;
      
      % AL: see waitforPoll() util
      % AL: see also aws ec2 wait, which sort of works but seems not
      % reliably
      starttime = tic;
      tfsucc = false;
      while true,
        [tf,state1] = obj.getInstanceState();
        if tf && strcmpi(state1,'running'),
          [tfexist,tfrun] = obj.inspectInstance();
          if tfexist && tfrun
            fprintf('... instance is started, waiting for full spin-up\n');
            % AL: aws ec2 wait instance-status-ok sort of worked, sort of
            starttime = tic;
            nAttempts = 0;
            while true,
              [st,res] = obj.runBatchCommandOutsideContainer('cat /dev/null');
              tfsucc = (st==0) ;
              if tfsucc,
                nAttempts = nAttempts + 1;
                fprintf('Attempt %d to connect to AWS EC2 instance succeeded!\n',nAttempts);
                break;
              else
                nAttempts = nAttempts + 1;
                fprintf('Attempt %d to connect to AWS EC2 instance failed: %s.\n',nAttempts,res);
                if toc(starttime) > maxwaittime,
                  break;
                end
              end
            end
            %tfsucc = waitforPoll(pollCbk,iterwaittime,maxwaittime);
            if tfsucc
              break;
            else
              return;
            end
          end
        end
        if toc(starttime) > maxwaittime,
          return;
        end
        pause(iterwaittime);
      end
    end  % function
    
    function errorIfInstanceNotRunning(obj)
      [doesInstanceExist, isInstanceRunning] = obj.inspectInstance() ;
      if ~doesInstanceExist
        error('EC2 instance with id %s does not seem to exist', obj.instanceID_);
      end
      if ~isInstanceRunning
        error('EC2 instance with id %s is not in the ''running'' state.',...
              obj.instanceID_)
      end
    end  % function
    
    function codestr = createShutdownAlarmCmd(obj,varargin)
      [periodsec,threshpct,evalperiods] = myparse(varargin,...
        'periodsec',300, ... % 5 mins
        'threshpct',5, ...
        'evalperiods',24 ... % 24x5mins=2hrs
      );
    
      % AL 20190213: Superficially pretty poor/confusing spec/api/doc for 
      % CloudWatch alarms. CloudWatch Dash is pretty suboptimal too. Maybe 
      % it's actually really smart (maybe not).
      %
      % 1. Prob don't want to go period<5min -- acts funny, maybe 
      % considered "high resolution" etc and requires special treatment.
      % 2. Prob don't want period much greater than 5 min -- b/c you want
      % the instance/cpu spin-up to get the CPUutil above the thresh in the
      % first interval. If the interval is too big then the spinup may not
      % get you above that threshold. Then, due to weird history retention
      % of alarms, you may just get repeated ALARM states and kills of your
      % instance.
      % 3. Deleting an alarm and bringing it back (with the same name) 
      % doesn't remove the history, and in some ways the triggers/criteria 
      % ignore the "missing gap" in time as if that interval (when the
      % alarm was deleted) was just removed from the space-time continuum.
      
      assert(periodsec==round(periodsec),'Expected integral periodsec.');
      
      [tfe,~,js] = obj.inspectInstance();
      if ~tfe
        error('AWS ec2 instance does not exist.');
      end
      
      availzone = js.Reservations.Instances.Placement.AvailabilityZone;
      regioncode = availzone(1:end-1);
      
      instanceID = obj.instanceID_;
      namestr = sprintf(obj.autoShutdownAlarmNamePat);
      descstr = sprintf('"Auto shutdown %d sec (%d periods)"',periodsec,evalperiods);
      dimstr = sprintf('"Name=InstanceId,Value=%s"',instanceID);
      arnstr = sprintf('arn:aws:automate:%s:ec2:stop',regioncode);
      
      code = {...
        'aws' 'cloudwatch' 'put-metric-alarm' ...
        '--alarm-name' namestr ...
        '--alarm-description' descstr ...
        '--metric-name' 'CPUUtilization' ...
        '--namespace' 'AWS/EC2' ...
        '--statistic' 'Average' ...
        '--period' num2str(periodsec) ...
        '--threshold' num2str(threshpct) ...
        '--comparison-operator' 'LessThanThreshold' ...
        '--dimensions' dimstr ...
        '--evaluation-periods' num2str(evalperiods) ...
        '--alarm-actions' arnstr ...
        '--unit' 'Percent' ...
        '--treat-missing-data' 'notBreaching' ...
        };
      codestr = String.cellstr2DelimList(code,' ');      
    end
    
    function [tfsucc,isalarm,reason] = checkShutdownAlarm(obj)
      tfsucc = false;
      isalarm = false;
      reason = '';
      if ~obj.isInstanceIDSet,
        reason = 'AWS EC2 instance not specified.';
        return;
      end
      namestr = sprintf(obj.autoShutdownAlarmNamePat);

      codestr = sprintf('aws cloudwatch describe-alarms --alarm-names "%s"',namestr);
      [st,json] = obj.syscmd(codestr,...
                             'failbehavior','warn',...
                             'isjsonout',true);
      tfsucc = (st==0) ;
      if ~tfsucc,
        reason = 'AWS CLI error calling describe-alarms.';
        return
      end
      json = jsondecode(json);
      if isempty(json) || isempty(json.MetricAlarms),
        tfsucc = true;
        return;
      end
      for i = 1:numel(json.MetricAlarms),
          j = find(strcmpi({json.MetricAlarms(i).Dimensions.Name},'InstanceId'),1);
          if ~isempty(j) && strcmp(json.MetricAlarms(i).Dimensions(j).Value,obj.instanceID_),
            isalarm = true;
            break;
          end
      end
      tfsucc = true;
    end
    
    function tfsucc = stopAlarm(obj)
      [tfsucc,isalarm,reason] = obj.checkShutdownAlarm();
      if ~tfsucc,
        warning('Could not check for alarm: %s\n',reason);
        return;
      end
      if ~isalarm,
        return;
      end
    end
    
    function tfsucc = configureAlarm(obj)
      % Note: this creates/puts a metricalarm with name based on the
      % instanceID. Currently the AWS API allows you to create the same
      % alarm multiple times with no harm (only one alarm is ultimately
      % created).

      [tfsucc,isalarm,reason] = obj.checkShutdownAlarm();
      if ~tfsucc,
        warning('Could  not check for alarm: %s',reason);
        return
      end
      % DEBUGAWS: AWS alarms are annoying while debugging
      if obj.isInDebugMode_ ,
        return
      end
      if isalarm,
        return
      end
      
      codestr = obj.createShutdownAlarmCmd() ;
      
      fprintf('Setting up AWS CloudWatch alarm to auto-shutdown your instance if it is idle for too long.\n');
      
      st = AWSec2.syscmd(codestr,...
                         'failbehavior','warn');
      tfsucc = (st==0) ;
    end
    
    % function tfsucc = getRemotePythonPID(obj)
    %   [st,res] = obj.runBatchCommandOutsideContainer('pgrep --uid ubuntu --oldest python');
    %   tfsucc = (st==0) ;
    %   if tfsucc
    %     pid = strtrim(res) ;
    %     obj.remotePID = pid; % right now each aws instance only has one GPU, so can only do one train/track at a time
    %     fprintf('Remote PID is: %s.\n\n',pid);
    %   else
    %     warningNoTrace('Failed to ascertain remote PID.');
    %   end
    % end
    
    function tfnopyproc = getNoPyProcRunning(obj)
      % Return true if there appears to be no python process running on
      % instance
      [st,res] = obj.runBatchCommandOutsideContainer('pgrep --uid ubuntu --oldest python',...
                                                     'failbehavior','silent');
      tfsucc = (st==0) ;
        
      % AL 20200213 First clause here is legacy: "expect command to fail; 
      % fail -> py proc killed". Running today on win10, the cmd always
      % succeeds whether a py proc is present or not. In latter case,
      % result is empty.
      tfnopyproc = ~tfsucc || isempty(res);
    end
 
    % FUTURE: use rsync if avail. win10 can ask users to setup WSL
    
    % function tfsucc = scpDownloadOrVerify(obj,srcAbs,dstAbs,varargin)
    %   % If dstAbs already exists, does NOT check identity of file against
    %   % dstAbs. In many cases, naming/immutability of files (with paths)
    %   % means this is OK.
    % 
    %   [sysCmdArgs] = ...
    %     myparse(varargin,...
    %             'sysCmdArgs',{}) ;
    % 
    %   if exist(dstAbs,'file') ,
    %     fprintf('File %s exists, not downloading.\n',dstAbs);
    %     tfsucc = true;
    %   else
    %     %logger.log('AWSSec2::scpDownloadOrVerify(): obj.scpcmd is %s\n', obj.scpCmd) ;
    %     cmd = AWSec2.scpDownloadCmd(obj.pem, obj.instanceIP, srcAbs, dstAbs, ...
    %                                 'scpcmd', obj.scpCmd) ;
    %     %logger.log('AWSSec2::scpDownloadOrVerify(): cmd is %s\n', cmd) ;
    %     st = AWSec2.syscmd(cmd,sysCmdArgs{:});
    %     tfsucc0 = (st==0) ;        
    %     tfsucc = tfsucc0 && logical(exist(dstAbs,'file')) ;
    %   end
    % end  % function
    
    % function tfsucc = scpDownloadOrVerifyEnsureDir(obj,srcAbs,dstAbs,varargin)
    %   dirLcl = fileparts(dstAbs);
    %   if exist(dirLcl,'dir')==0
    %     [tfsucc,msg] = mkdir(dirLcl);
    %     if ~tfsucc
    %       warningNoTrace('Failed to create local directory %s: %s',dirLcl,msg);
    %       return;
    %     end
    %   end
    %   tfsucc = obj.scpDownloadOrVerify(srcAbs,dstAbs,varargin{:});
    % end
 
    % function tfsucc = scpUpload(obj,file,dest,varargin)
    %   [destRelative,sysCmdArgs] = myparse(varargin,...
    %     'destRelative',true,... % true if dest is relative to ~
    %     'sysCmdArgs',{});
    %   cmd = AWSec2.scpPrepareUploadCmd(obj.pem,obj.instanceIP,dest,...
    %                                    'destRelative',destRelative);
    %   AWSec2.syscmd(cmd,sysCmdArgs{:});
    %   cmd = AWSec2.scpUploadCmd(file,obj.pem,obj.instanceIP,dest,...
    %                             'scpcmd',obj.scpCmd,'destRelative',destRelative);
    %   st = AWSec2.syscmd(cmd,sysCmdArgs{:});
    %   tfsucc = (st==0) ;
    % end
    
    % function scpUploadOrVerify(obj,src,dst,fileDescStr,varargin) % throws
    %   % Either i) confirm a remote file exists, or ii) upload it.
    %   % In the case of i), NO CHECK IS MADE that the existing file matches
    %   % the local file.
    %   %
    %   % Could use rsync here instead but rsync is less likely to be 
    %   % installed on a Win machine
    %   %
    %   % This method either succeeds or fails and harderrors.
    %   %
    %   % src: full path to local file
    %   % dstRel: relative (to home) path to destination
    %   % fileDescStr: eg 'training file' or 'movie'
    % 
    %   destRelative = myparse(varargin,...
    %     'destRelative',true);
    % 
    %   if destRelative
    %     dstAbs = ['~/' dst];
    %   else
    %     dstAbs = dst;
    %   end
    % 
    %   src_d = dir(src);
    %   src_sz = src_d.bytes;
    %   tfsucc = obj.fileExistsAndIsGivenSize(dstAbs,src_sz);
    %   if tfsucc
    %     fprintf('%s file exists: %s.\n\n',...
    %       String.niceUpperCase(fileDescStr),dstAbs);
    %   else
    %     %obj.SetStatus(sprintf('Uploading %s file to AWS EC2 instance',fileDescStr));
    %     fprintf('About to upload. This could take a while depending ...\n');
    %     tfsucc = obj.scpUpload(src,dstAbs,...
    %                            'destRelative',false,'sysCmdArgs',{});
    %     %obj.ClearStatus();
    %     if tfsucc
    %       fprintf('Uploaded %s %s to %s.\n\n',fileDescStr,src,dst);
    %     else
    %       error('Failed to upload %s %s.',fileDescStr,src);
    %     end
    %   end
    % end
    
    % function scpUploadOrVerifyEnsureDir(obj,fileLcl,fileRemote,fileDescStr,...
    %     varargin)
    %   % Upload a file to a dir which may not exist yet. Create it if 
    %   % necessary. Either succeeds, or fails and harderrors.
    % 
    %   destRelative = myparse(varargin,...
    %                          'destRelative',false);
    % 
    %   dirRemote = fileparts(fileRemote);
    %   obj.ensureRemoteDirExists(dirRemote,'relative',destRelative); 
    %   obj.scpUploadOrVerify(fileLcl,fileRemote,fileDescStr,...
    %     'destRelative',destRelative); % throws
    % end
    
    function rsyncUploadFolder(obj, srcWslPath, destRemotePath)
      % rsync the local folder src to the remote folder dest.  No local-to-remote
      % translation is done on dest.  Thus src should be a WSL path, dest should be
      % a remote path.
      % This can throw APT:syscmd error.       
      cmd = AWSec2.rsyncUploadFolderCmd(srcWslPath, obj.pem, obj.instanceIP, destRemotePath) ;
      AWSec2.syscmd(cmd, 'failbehavior', 'err') ;
    end

    function rsyncDownloadFolder(obj, srcRemotePath, destWslPath)
      % This can throw APT:syscmd error.       

      % Create the parent dir for the destination file
      destParentFolderWslPath = linux_fileparts2(destWslPath) ;
      ensureWslFolderExists(destParentFolderWslPath) ;

      % Do the rsync command
      cmd = AWSec2.rsyncDownloadFolderCmd(srcRemotePath, obj.pem, obj.instanceIP, destWslPath) ;
      AWSec2.syscmd(cmd, 'failbehavior', 'err') ;
    end

    function rsyncDownloadFile(obj, srcRemotePath, destWslPath)
      % rsync the local folder src to the remote folder dest.  No local-to-remote
      % translation is done on dest.
      % This can throw APT:syscmd error.

      % Create the parent dir for the destination file
      destParentFolderWslPath = linux_fileparts2(destWslPath) ;
      ensureWslFolderExists(destParentFolderWslPath) ;

      % Do the rsync command
      cmd = AWSec2.rsyncDownloadFileCmd(obj.pem, obj.instanceIP, srcRemotePath, destWslPath) ;
      AWSec2.syscmd(cmd, 'failbehavior', 'err') ;
    end

    function rsyncUploadFile(obj, srcWslPath, destRemotePath)
      % rsync the local folder src to the remote folder dest.  No local-to-remote
      % translation is done on dest.  Thus src should be a local WSL path, and dest
      % should be a remote path.
      % This can throw APT:syscmd error.       
      cmd = AWSec2.rsyncUploadFileCmd(srcWslPath, obj.pem, obj.instanceIP, destRemotePath) ;
      AWSec2.syscmd(cmd, 'failbehavior', 'err') ;
    end

    function deleteFile(obj,dst,~,varargin)
      % Either i) confirm a remote file does not exist, or ii) deletes it.
      % This method either succeeds or fails and harderrors.
      %
      % dst: path to file on remote system
      % dstRel: relative (to home) path to destination
      % fileDescStr: eg 'training file' or 'movie'
      
      destRelative = myparse(varargin,...
        'destRelative',true);
      
      if destRelative
        if iscell(dst),
          dstAbs = cellfun(@(x) ['~/' x],dst,'Uni',0);
        else
          dstAbs = ['~/' dst];
        end
      else
        dstAbs = dst;
      end
      
      if iscell(dstAbs),
        cmd = ['rm -f',sprintf(' "%s"',dstAbs{:})];
      else
        cmd = sprintf('rm -f "%s"',dstAbs);
      end
      %obj.SetStatus(sprintf('Deleting %s file(s) (if they exist) from AWS EC2 instance',fileDescStr));
      obj.runBatchCommandOutsideContainer(cmd,'failbehavior','err');
      %obj.ClearStatus();
    end    
    
    function tf = fileExists(obj, wsl_file_path)
      %script = '/home/ubuntu/APT/matlab/misc/fileexists.sh';
      %cmdremote = sprintf('%s %s',script,f);
      cmdremote = sprintf('/usr/bin/test -e %s ; echo $?',wsl_file_path);
      [~,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err');  % will handle WSL->remote file path substitution
      tf = strcmp(strtrim(res),'0') ;      
    end
    
    function tf = fileExistsAndIsNonempty(obj, wsl_file_path)
      %script = '/home/ubuntu/APT/matlab/misc/fileexistsnonempty.sh';
      %cmdremote = sprintf('%s %s',script,f);
      cmdremote = sprintf('/usr/bin/test -s %s ; echo $?',wsl_file_path);
      [~,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err');  % will handle WSL->remote file path substitution
      tf = strcmp(strtrim(res),'0') ;      
    end
    
    function tf = fileExistsAndIsGivenSize(obj, wsl_file_path, query_byte_count)
      script_path = '/home/ubuntu/APT/matlab/misc/fileexists.sh';
      cmdremote = sprintf('%s %s %d', script_path, wsl_file_path, query_byte_count) ;
      %cmdremote = sprintf('/usr/bin/stat --printf="%%s\\n" %s',f) ;  
        % stat return code is nonzero if file is missing---annoying
      [~, res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err');
      actual_byte_count = str2double(res) ;
      tf = (actual_byte_count == query_byte_count) ;
    end
    
    function s = fileContents(obj, wsl_file_path, varargin)
      % First check if the file exists
      cmdremote = sprintf('/usr/bin/test -e %s ; echo $?',wsl_file_path);
      [st,res] = obj.runBatchCommandOutsideContainer(cmdremote, 'failbehavior', 'silent', varargin{:}) ;
      if st==0 ,
        % Command succeeded in determining whether the file exists
        if strcmp(strtrim(res),'0') ,
          % File exists
          cmdremote = sprintf('cat %s',wsl_file_path);
          [st,res] = obj.runBatchCommandOutsideContainer(cmdremote, 'failbehavior', 'silent', varargin{:}) ;
          if st==0
            s = res ;
          else
            s = sprintf('<Unable to read file: %s>', res) ;
          end
        else
          % File doesn't exist
          s = '<file does not exist>' ;          
        end
      else
        % Command failed while trying to determine if the file exists
        s = sprintf('<Unable to determine if file exists: %s>', res) ;
      end
    end  % function
    
    function result = remoteFileModTime(obj, wsl_file_path, varargin)
      % Returns the file modification time (mtime) in seconds since Epoch
      command = sprintf('stat --format=%%Y %s', escape_string_for_bash(wsl_file_path)) ;  % time of last data modification, seconds since Epoch
      [st, stdouterr] = obj.runBatchCommandOutsideContainer(command, varargin{:}) ; 
      did_succeed = (st==0) ;
      if did_succeed ,
        result = str2double(stdouterr) ;
      else
        % Warning/error happens inside obj.runBatchCommandOutsideContainer(), so just set a fallback value
        result = nan ;
      end
    end
    
    function tfsucc = lsdir(obj, wsl_dir_path, varargin)
      [failbehavior,args] = myparse(varargin,...
                                    'failbehavior','warn',...
                                    'args','-lha') ;      
      cmdremote = sprintf('ls %s %s', args, wsl_dir_path);
      [st,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior',failbehavior);
      tfsucc = (st==0) ;
      disp(res);
      % warning thrown etc per failbehavior
    end
    
    % function remoteDirFull = ensureRemoteDirExists(obj,remoteDir,varargin)
    %   % Creates/verifies remote dir. Either succeeds, or fails and harderrors.
    % 
    %   [relative,descstr] = myparse(varargin,...
    %     'relative',true,...  % true if remoteDir is relative to ~
    %     'descstr',''... % cosmetic, for disp/err strings
    %     );
    % 
    %   if ~isempty(descstr)
    %     descstr = [descstr ' '];
    %   end
    % 
    %   if relative
    %     remoteDirFull = ['~/' remoteDir];
    %   else
    %     remoteDirFull = remoteDir;
    %   end
    % 
    %   cmdremote = sprintf('mkdir -p %s',remoteDirFull);
    %   [st,res] = obj.runBatchCommandOutsideContainer(cmdremote);
    %   tfsucc = (st==0) ;
    %   if tfsucc
    %     fprintf('Created/verified remote %sdirectory %s: %s\n\n',...
    %       descstr,remoteDirFull,res);
    %   else
    %     error('Failed to create remote %sdirectory %s: %s',descstr,...
    %       remoteDirFull,res);
    %   end
    % end
    
    function remotePaths = remoteGlob_(obj, wslGlobs)
      % Look for remote files/paths. Either succeeds, or fails and harderrors.

      % globs: cellstr of globs
      
      lscmdAsList = cellfun(@(glob)sprintf('ls %s 2> /dev/null ; ',glob), wslGlobs, 'uni', 0) ;
      lscmd = cat(2,lscmdAsList{:});
      [st,res] = obj.runBatchCommandOutsideContainer(lscmd);
      tfsucc = (st==0) ;      
      if tfsucc
        remotePaths = regexp(res,'\n','split');
        remotePaths = remotePaths(:);
        remotePaths = remotePaths(~cellfun(@isempty,remotePaths));
      else
        error('Failed to find remote files/paths %s: %s',...
              String.cellstr2CommaSepList(wslGlobs),res);
      end      
    end  % function
    
    function result = wrapCommandSSH(obj, input_command, varargin)
      % Wrap input command to run on the remote host via ssh.  Performs WSL-> remote
      % path substition unless optional dopathsubs argsument is false.
      dopathsubs = ...
        myparse(varargin, ...
                'dopathsubs', true) ;
      if dopathsubs
        remote_command = obj.applyFilePathSubstitutions(input_command) ;
      else
        remote_command = input_command ;
      end
      result = wrapCommandSSH(remote_command, ...
                              'host', obj.instanceIP, ...
                              'timeout',8, ...
                              'username', 'ubuntu', ...
                              'identity', obj.pem) ;
    end  % function

    function [st,res] = runBatchCommandOutsideContainer(obj, cmdremote, varargin)      
      % Runs a single command-line command on the ec2 instance.
      % Performs WSL-> remote path substition.

      % Determine whether obj.wrapCommandSSH should do WSL->remote path
      % substitutions
      [dopathsubs, ...
       restOfVarargin] = ...
        myparse_nocheck(varargin, ...
                        'dopathsubs', true) ;

      % Wrap for ssh'ing into an AWS instance
      cmd1 = obj.wrapCommandSSH(cmdremote, 'dopathsubs', dopathsubs) ;  % uses fields of obj to set parameters for ssh command
    
      % Need to prepend a sleep to avoid problems
      precommand = 'sleep 5 && export AWS_PAGER=' ;
        % Change the sleep value at your peril!  I changed it to 3 and everything
        % seemed fine for a while, until it became a very hard-to-find bug!  
        % --ALT, 2024-09-12
      command = sprintf('%s && %s', precommand, cmd1) ;

      % Issue the command, gather results
      [st, res] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, restOfVarargin{:}) ;      
    end
        
%     function cmd = sshCmdGeneralLogged(obj, cmdremote, logfileremote)
%       cmd = sprintf('%s -i %s ubuntu@%s "%s </dev/null >%s 2>&1 &"',...
%                     obj.sshCmd, obj.pem, obj.instanceIP, cmdremote, logfileremote) ;
%     end
        
    % function tf = canKillRemoteProcess(obj)
    %   tf = ~isempty(obj.remotePID) ;
    % end
    
    function killRemoteProcess(obj)
      % Just kill all the Python processes on the EC2 instance
      cmdremote = 'pkill --uid ubuntu --full python';
      [st,~] = obj.runBatchCommandOutsideContainer(cmdremote);
      if st==0 ,
        fprintf('Kill command sent.\n\n');
      else
        error('Kill command failed.');
      end
    end  % function
  end  % methods
  
  methods (Static)
    
    function cmd = launchInstanceCmd(keyName, varargin)
      [ami,instType,secGrp,dryrun] = myparse(varargin,...
        'ami',AWSec2.AMI,...
        'instType',AWSec2.instanceType,...
        'secGrp',AWSec2.AWS_SECURITY_GROUP,...
        'dryrun',false);
      date_and_time_string = char(datetime('now','TimeZone','local','Format','yyyy-MM-dd-HH-mm-ss')) ;
      name = sprintf('apt-to-the-porpoise-%s', date_and_time_string) ;
      tag_specifications = sprintf('ResourceType=instance,Tags=[{Key=Name,Value=%s}]', name) ;
      escaped_tag_specifications = escape_string_for_bash(tag_specifications) ;
      block_device_mapping = '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":800,"DeleteOnTermination":true,"VolumeType":"gp3"}}]' ;
      escaped_block_device_mapping = escape_string_for_bash(block_device_mapping) ;
      cmd_template = ...
        strcatg('aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s ', ...
                '--tag-specifications %s --block-device-mappings %s') ;
      cmd = sprintf(cmd_template, ami, instType, secGrp, escaped_tag_specifications, escaped_block_device_mapping) ;
      if dryrun,
        cmd = strcatg(cmd,' --dry-run') ;
      end
      if ~isempty(keyName),
        cmd = strcatg(cmd, ' --key-name ', keyName) ;
      end
    end  % function
    
    function cmd = listInstancesCmd(keyName,varargin)      
      [ami,instType,secGrp] = ...
        myparse(varargin,...
                'ami',AWSec2.AMI,...
                'instType','p3.2xlarge',...
                'secGrp',AWSec2.AWS_SECURITY_GROUP,...
                'dryrun',false);
      
      cmd = sprintf('aws ec2 describe-instances --filters "Name=image-id,Values=%s" "Name=instance.group-name,Values=%s" "Name=key-name,Values=%s"', ...
                    ami,secGrp,keyName);
      if ~isempty(instType)
        cmd = [cmd sprintf(' "Name=instance-type,Values=%s"',instType)];
      end
    end
    
    function cmd = describeInstancesCmd(ec2id)
      cmd = sprintf('aws ec2 describe-instances --instance-ids %s',ec2id);
    end
    
    function cmd = stopInstanceCmd(ec2id)
      cmd = sprintf('aws ec2 stop-instances --instance-ids %s',ec2id);
    end

    function cmd = startInstanceCmd(ec2id)
      cmd = sprintf('aws ec2 start-instances --instance-ids %s',ec2id);
    end
    
    function cmd = scpPrepareUploadCmd(pem,ip,dest,varargin)
      destRelative = myparse(varargin,...
                             'destRelative',true);
      if destRelative
        dest = linux_fullfile('~',dest);
      end
      parentdir = fileparts(dest);
      cmdremote = sprintf('mkdir -p %s',parentdir);
      cmd = wrapCommandSSH(cmdremote, ...
                           'host', ip, ...
                           'timeout',8, ...
                           'username', 'ubuntu', ...
                           'identity', pem) ;      
    end
    
    function cmd = scpUploadCmd(file,pem,ip,dest,varargin)
      [destRelative,scpcmd] = myparse(varargin,...
                                      'destRelative',true,...
                                      'scpcmd','scp');
      if destRelative
        dest = ['~/' dest];
      end
      if ispc() 
        [fileP,fileF,fileE] = fileparts(file);
        % 20190501. scp on windows is dumb and treats colons ':' as a
        % host specifier etc. there may not be a good way to escape; 
        % googling says use pscp or other scp impls. 
        
        fileP = regexprep(fileP,'/','\\');
        fileF = regexprep(fileF,'/','\\');
        cmd = sprintf('pushd %s && %s -i %s %s ubuntu@%s:%s',fileP,scpcmd,...
          pem,['.\' fileF fileE],ip,dest); % fileP here can contain a space and pushd will do the right thing!
      else
        cmd = sprintf('%s -i %s %s ubuntu@%s:%s',scpcmd,pem,file,ip,dest);
      end
    end

    function cmd = scpDownloadCmd(pem, ip, srcAbs, dstAbs, varargin)
      scpcmd = myparse(varargin,...
                       'scpcmd', 'scp') ;
      cmd = sprintf('%s -i %s -r ubuntu@%s:"%s" "%s"',scpcmd,pem,ip,srcAbs,dstAbs);
    end

    function cmd = rsyncDownloadFileCmd(pemFilePath, ip, srcFileRemotePath, destFileWslPath)
      % Generate the system() command to download a file via rsync.

      % It's important that neither src nor dest have a trailing slash
      if isempty(srcFileRemotePath) ,
        error('src file for rsync cannot be empty') ;
      else
        if strcmp(srcFileRemotePath(end),'/') ,
          error('src file for rsync cannot end in a slash') ;
        end
      end
      if isempty(destFileWslPath) ,
        error('dest file for rsync cannot be empty') ;
      else
        if strcmp(destFileWslPath(end),'/') ,
          error('dest file for rsync cannot end in a slash') ;
        end
      end

      % Generate the --rsh argument
      %sshcmd = sprintf('%s -o ConnectTimeout=8 -i %s', AWSec2.sshCmd, pemFilePath) ;
      sshcmd = wrapCommandSSH('', 'host', '', 'timeout', 8, 'identity', pemFilePath) ;
        % We use an empty command, and an empty host, to get a string with the default
        % options plus the two options we want to specify.
      escaped_sshcmd = escape_string_for_bash(sshcmd) ;

      % Generate the final command
      cmd = sprintf('%s --rsh=%s ubuntu@%s:%s %s', AWSec2.rsyncCmd, escaped_sshcmd, ip, srcFileRemotePath, destFileWslPath) ;
    end

    function cmd = rsyncUploadFileCmd(srcFileWslPath, pemFilePath, ip, destFileRemotePath)
      % Generate the system() command to upload a file via rsync.

      % It's important that neither src nor dest have a trailing slash
      if isempty(srcFileWslPath) ,
        error('src file for rsync cannot be empty') ;
      else
        if strcmp(srcFileWslPath(end),'/') ,
          error('src file for rsync cannot end in a slash') ;
        end
      end
      if isempty(destFileRemotePath) ,
        error('dest file for rsync cannot be empty') ;
      else
        if strcmp(destFileRemotePath(end),'/') ,
          error('dest file for rsync cannot end in a slash') ;
        end
      end

      % Generate the --rsh argument
      sshcmd = wrapCommandSSH('', 'host', '', 'timeout', 8, 'identity', pemFilePath) ;
        % We use an empty command, and an empty host, to get a string with the default
        % options plus the two options we want to specify.
      escaped_sshcmd = escape_string_for_bash(sshcmd) ;

      % Generate the final command
      cmd = sprintf('%s --rsh=%s %s ubuntu@%s:%s', AWSec2.rsyncCmd, escaped_sshcmd, srcFileWslPath, ip, destFileRemotePath) ;
    end

    function cmd = rsyncUploadFolderCmd(srcWslPath, pemFilePath, ip, destRemotePath)
      % Generate the system() command to upload a folder via rsync.

      % It's important that neither src nor dest have a trailing slash
      if isempty(srcWslPath) ,
        error('src folder for rsync cannot be empty') ;
      else
        if strcmp(srcWslPath(end),'/') ,
          error('src folder for rsync cannot end in a slash') ;
        end
      end
      if isempty(destRemotePath) ,
        error('dest folder for rsync cannot be empty') ;
      else
        if strcmp(destRemotePath(end),'/') ,
          error('dest folder for rsync cannot end in a slash') ;
        end
      end

      % Generate the --rsh argument
      %sshcmd = sprintf('%s -o ConnectTimeout=8 -i %s', AWSec2.sshCmd, pemFilePath) ;
      sshcmd = wrapCommandSSH('', 'host', '', 'timeout', 8, 'identity', pemFilePath) ;
        % We use an empty command, and an empty host, to get a string with the default
        % options plus the two options we want to specify.
      escaped_sshcmd = escape_string_for_bash(sshcmd) ;

      % Generate the final command
      cmd = sprintf('%s --rsh=%s %s/ ubuntu@%s:%s', AWSec2.rsyncCmd, escaped_sshcmd, srcWslPath, ip, destRemotePath) ;
    end
    
    function cmd = rsyncDownloadFolderCmd(srcRemotePath, pemFilePath, ip, destWslPath)
      % Generate the system() command to upload a folder via rsync.

      % It's important that neither src nor dest have a trailing slash
      if isempty(srcRemotePath) ,
        error('src folder for rsync cannot be empty') ;
      else
        if strcmp(srcRemotePath(end),'/') ,
          error('src folder for rsync cannot end in a slash') ;
        end
      end
      if isempty(destWslPath) ,
        error('dest folder for rsync cannot be empty') ;
      else
        if strcmp(destWslPath(end),'/') ,
          error('dest folder for rsync cannot end in a slash') ;
        end
      end

      % Generate the --rsh argument
      %sshcmd = sprintf('%s -o ConnectTimeout=8 -i %s', AWSec2.sshCmd, pemFilePath) ;
      sshcmd = wrapCommandSSH('', 'host', '', 'timeout', 8, 'identity', pemFilePath) ;
        % We use an empty command, and an empty host, to get a string with the default
        % options plus the two options we want to specify.
      escaped_sshcmd = escape_string_for_bash(sshcmd) ;

      % Generate the final command
      cmd = sprintf('%s --rsh=%s ubuntu@%s:%s/ %s/', AWSec2.rsyncCmd, escaped_sshcmd, ip, srcRemotePath, destWslPath) ;
    end
    
%     function cmd = sshCmdGeneral(sshcmd, pem, ip, cmdremote, varargin)
%       [timeout,~] = myparse(varargin,...
%                             'timeout',8,...
%                             'usedoublequotes',false);
%       
%       args = { sshcmd '-i' pem sprintf('-o ConnectTimeout=%d', timeout) sprintf('ubuntu@%s',ip) } ;
%       args{end+1} = escape_string_for_bash(cmdremote) ;
%       cmd = space_out(args,' ');
%     end  % function

    % function scpCmd = computeScpCmd()
    %   if ispc()
    %     windows_null_device_path = '\\.\NUL' ;
    %     scpCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSCPCMD, windows_null_device_path) ; 
    %   else
    %     scpCmd = 'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
    %   end
    % end

%     function sshCmd = computeSshCmd()
%       if ispc()
%         windows_null_device_path = '\\.\NUL' ;
%         sshCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSSHCMD, windows_null_device_path) ; 
%       else
%         sshCmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
%       end
%     end
    
    function result = computeRsyncCmd()
      if ispc()  %#ok<IFBDUP>
        result = 'rsync -az' ;
      else
        result = 'rsync -az' ;
      end
    end
    
    function [st,res,warningstr] = syscmd(cmd0, varargin)      
      cmd = sprintf('sleep 5 && export AWS_PAGER= && %s', cmd0) ;
        % Change the sleep value at your peril!  I changed it to 3 and everything
        % seemed fine for a while, until it became a very hard-to-find bug!  
        % --ALT, 2024-09-12
      [st,res,warningstr] = apt.syscmd(cmd, varargin{:}) ;
    end  % function    
  end  % Static methods block
  
  methods
    function suitcase = packParfevalSuitcase(obj)
      % Use before calling parfeval, to restore Transient properties that we want to
      % survive the parfeval boundary.
      suitcase = struct() ;
      suitcase.instanceIP = obj.instanceIP ;
      suitcase.isInDebugMode_ = obj.isInDebugMode_ ;
      suitcase.isProjectCacheRemote_ = obj.isProjectCacheRemote_ ;
      suitcase.wslProjectCachePath_ = obj.wslProjectCachePath_ ;
      suitcase.didUploadMovies_ = obj.didUploadMovies_ ;
      suitcase.localPathFromMovieIndex_ = obj.wslPathFromMovieIndex_ ;
      suitcase.remotePathFromMovieIndex_ = obj.remotePathFromMovieIndex_ ;
    end  % function
    
    function restoreAfterParfeval(obj, suitcase)
      % Should be called in background tasks run via parfeval, to restore fields that
      % should not be restored from persistence, but we want to survive the parfeval
      % boundary.
      obj.instanceIP = suitcase.instanceIP ;
      obj.isInDebugMode_ = suitcase.isInDebugMode_ ;
      obj.isProjectCacheRemote_ = suitcase.isProjectCacheRemote_ ;
      obj.wslProjectCachePath_ = suitcase.wslProjectCachePath_ ;
      obj.didUploadMovies_ = suitcase.didUploadMovies_ ;
      obj.wslPathFromMovieIndex_ = suitcase.localPathFromMovieIndex_ ;
      obj.remotePathFromMovieIndex_ = suitcase.remotePathFromMovieIndex_ ;
    end  % function

    function downloadTrackingFilesIfNecessary(obj, pollingResult, wslProjectCachePath, movfiles)
      % Errors if something goes wrong.
      remoteProjectCachePath = AWSec2.remoteDLCacheDir ;
      currentLocalPathFromTrackedMovieIndex = movfiles(:) ;  % want cellstr col vector
      originalLocalPathFromTrackedMovieIndex = pollingResult.movfile(:) ;  % want cellstr col vector
      if all(strcmp(currentLocalPathFromTrackedMovieIndex,originalLocalPathFromTrackedMovieIndex))
        % we perform this check b/c while tracking has been running in
        % the bg, the project could have been updated, movies
        % renamed/reordered etc.        
        % download trkfiles 
        nativeLocalTrackFilePaths = pollingResult.trkfile ;
        wslLocalTrackFilePaths = wsl_path_from_native(nativeLocalTrackFilePaths) ;
        remoteTrackFilePaths = linux_replace_prefix_path(wslLocalTrackFilePaths, wslProjectCachePath, remoteProjectCachePath) ;
        % sysCmdArgs = {'failbehavior', 'err'};
        for ivw=1:numel(wslLocalTrackFilePaths)
          remoteTrackFilePath = remoteTrackFilePaths{ivw};
          wslLocalTrackFilePath = wslLocalTrackFilePaths{ivw};
          fprintf('Trying to download %s to %s...\n',remoteTrackFilePath,wslLocalTrackFilePath);
          % obj.scpDownloadOrVerifyEnsureDir(trkRmt,trkLcl,'sysCmdArgs',sysCmdArgs); % XXX doc orVerify
          obj.rsyncDownloadFile(remoteTrackFilePath, wslLocalTrackFilePath) ;
          fprintf('Done downloading %s to %s.\n',remoteTrackFilePath,wslLocalTrackFilePath);
        end
      else
        error('Tracking complete, but one or move movies has been changed in current project.') ;
          % conservative, take no action for now
      end
    end  % function    

    function result = get.isProjectCacheRemote(obj)
      result = obj.isProjectCacheRemote_ ;
    end  % function

    function result = get.isProjectCacheLocal(obj)
      result = ~obj.isProjectCacheRemote_ ;
    end  % function    

    function [tfsucc,res] = batchPoll(obj, wsl_fspollargs)
      % fspollargs: [n] cellstr eg {'exists' '/my/file' 'existsNE' '/my/file2'}
      % All paths should be WSL paths.
      %
      % res: [n] cellstr of fspoll responses

      assert(iscellstr(wsl_fspollargs) && ~isempty(wsl_fspollargs)) ;  %#ok<ISCLSTR> 
      fspollargsCount = numel(wsl_fspollargs) ;
      assert(mod(fspollargsCount,2)==0) ;  % has to be even
      responseCount = fspollargsCount/2 ;
      
      fspollstr = space_out(wsl_fspollargs);
      fspoll_script_path = '/home/ubuntu/APT/matlab/misc/fspoll.py' ;

      cmdremote = sprintf('%s %s',fspoll_script_path,fspollstr);

      [st,res] = obj.runBatchCommandOutsideContainer(cmdremote);  % will translate WSL paths to remote paths
      tfsucc = (st==0) ;
      if tfsucc
        res = regexp(res,'\n','split');
        tfsucc = iscell(res) && (numel(res)==responseCount+1) ;  % last cell is {0x0 char}
        res = res(1:end-1);
      else
        res = [];
      end
    end  % function
    
    function maxiter = getMostRecentModel(obj, dmc)  % constant method
      % Get the number of iterations completed for the model indicated by dmc.
      % Note that dmc will have native paths in it.
      % Also note that maxiter is in general a row vector.
      if obj.isProjectCacheRemote_ ,
        % maxiter is nan if something bad happened or if DNE
        % TODO allow polling for multiple models at once
        [dirModelChainLnx,idx] = dmc.dirModelChainLnx() ;
          % The first return arg from dmc.dirModelChainLnx() is a cellstring of paths.
          % In spite of the method name, the paths are *native*.
        fspollargs = {};
        for i = 1:numel(idx),
          native_path = dirModelChainLnx{i} ;
          wsl_path = wsl_path_from_native(native_path) ;
          fspollargs = horzcat(fspollargs, {'mostrecentmodel', wsl_path} ) ;  %#ok<AGROW>
        end
        [tfsucc, res] = obj.batchPoll(fspollargs) ;
        if tfsucc
          maxiter = str2double(res(1:numel(idx))); % includes 'DNE'->nan
        else
          maxiter = nan(1,numel(idx));
        end        
      else
        maxiter = DLBackEndClass.getMostRecentModelLocal_(dmc) ;
      end
    end  % function
    
    function [didsucceed, msg] = mkdir(obj, wsl_dir_path)
      % Create the named directory on the remote AWS machine.      
      quoted_dirloc = escape_string_for_bash(wsl_dir_path) ;
      base_command = sprintf('mkdir -p %s', quoted_dirloc) ;
      [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;  % Will translate to remote path
      didsucceed = (status==0) ;
    end
    
    function ensureRemoteFolderExists(obj, remote_dir_path)
      % Create the named directory on the remote AWS machine.  Note that this does
      % *not* do WSL->remote path translation, and also that it *throws* if it is
      % unable to perform its duties.  It does not return anything.
      quoted_remote_dir_path = escape_string_for_bash(remote_dir_path) ;
      base_command = sprintf('mkdir -p %s', quoted_remote_dir_path) ;
      obj.runBatchCommandOutsideContainer(base_command, 'dopathsubs', false, 'failbehavior', 'err') ;
    end
    
    function uploadProjectCacheIfNeeded(obj, nativeProjectCachePath)
      % Take a local DMC and mirror/upload it to the AWS instance aws; 
      % update .rootDir, .reader appropriately to point to model on remote 
      % disk.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the remote aws instance.
      %
      % PostConditions: 
      % - remote cachedir mirrors this model for key model files; "extra"
      % remote files not removed; identities of existing files not
      % confirmed but naming/immutability of DL artifacts makes this seem
      % safe
      % - .rootDir updated to remote cacheloc
      % - .reader update to AWS reader
      
      % Sanity checks
      % assert(isa(dmc, 'DeepModelChainOnDisk')) ;      
      % assert(isscalar(dmc));

      % If the DMC is already remote, do nothing
      if obj.isProjectCacheRemote ,
        return
      end

      % % Make sure there is a trained model
      % maxiter = obj.getMostRecentModel(dmc) ;
      % succ = (maxiter >= 0) ;
      % if strcmp(mode, 'tracking') && any(~succ) ,
      %   dmclfail = dmc.dirModelChainLnx(find(~succ));
      %   fstr = sprintf('%s ',dmclfail{:});
      %   error('Failed to determine latest model iteration in %s.',fstr);
      % end
      % if isnan(maxiter) ,
      %   fprintf('Currently, there is no trained model.\n');
      % else
      %   fprintf('Current model iteration is %s.\n',mat2str(maxiter));
      % end
     
      % Make sure there is a live backend
      obj.errorIfInstanceNotRunning();  % throws error if ec2 instance is not connected
      
      % Sync remote /home/ubuntu/cacheDL from ~/.apt/tpwhatever_blah_blah_blah
      wslProjectCachePath = wsl_path_from_native(nativeProjectCachePath) ;
      obj.wslProjectCachePath_ = wslProjectCachePath ;  % Need to set this before calling obj.remote_path_from_wsl()
      remoteProjectCachePath = AWSec2.remoteDLCacheDir ;
      [didsucceed, msg] = obj.mkdir(remoteProjectCachePath) ;
      if ~didsucceed ,
        error('Unable to create remote dir %s.\nmsg:\n%s\n', remoteProjectCachePath, msg) ;
      end
      obj.rsyncUploadFolder(wslProjectCachePath, remoteProjectCachePath) ;  % this will throw if there's a problem

      % If we made it here, upload successful---update the state to reflect that the
      % model is now remote.      
      obj.isProjectCacheRemote_ = true ;
    end  % function
    
    function downloadProjectCacheIfNeeded(obj, nativeProjectCachePath)
      % Inverse of mirror2remoteAws. Download/mirror model from remote AWS
      % instance to local cache.
      %
      % update .rootDir, .reader appropriately to point to model in local
      % cache.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the local cache.

      % If the DMC is already local,  do nothing
      if obj.isProjectCacheLocal ,
        return
      end
      
      % Make sure there is a live backend
      obj.errorIfInstanceNotRunning();  % throws error if ec2 instance is not connected

      % Download
      wslProjectCachePath = wsl_path_from_native(nativeProjectCachePath) ;
      obj.wslProjectCachePath_ = wslProjectCachePath ;  % Need to set this before calling obj.remote_path_from_wsl()
      remoteProjectCachePath = AWSec2.remoteDLCacheDir ;
      ensureWslFolderExists(wslProjectCachePath) ;  % will throw if fails
      obj.rsyncDownloadFolder(remoteProjectCachePath, wslProjectCachePath) ;  % this will throw if there's a problem
      
      % If we made it here, download successful---update the state to reflect that the
      % model is now local.            
      obj.isProjectCacheRemote_ = false ;
    end  % function
        
    % function result = getTorchHome(obj)
    %   if obj.isProjectCacheRemote_ ,
    %     result = linux_fullfile(AWSec2.remoteDLCacheDir, 'torch') ;
    %   else
    %     result = fullfile(APT.getdotaptdirpath(), 'torch') ;
    %   end
    % end  % function
    
    function result = get.wslProjectCachePath(obj) 
      result = obj.wslProjectCachePath_ ;
    end  % function

    function set.wslProjectCachePath(obj, value) 
      obj.wslProjectCachePath_ = value ;
    end  % function
    
    % function result = get.wslTorchCachePath(obj) 
    %   result = obj.wslTorchCachePath_ ;
    % end  % function

    % function result = get.remoteDMCRootDir(obj)
    %   result = AWSec2.remoteDLCacheDir ;
    % end  % function
        
    function uploadMovies(obj, wslPathFromMovieIndex)
      % Upload movies to the backend, if necessary.
      if obj.didUploadMovies_ ,
        return
      end
      obj.ensureRemoteFolderExists(AWSec2.remoteMovieCacheDir) ;  % throws if error
      remotePathFromMovieIndex = AWSec2.remote_movie_path_from_wsl(wslPathFromMovieIndex) ;
      movieCount = numel(wslPathFromMovieIndex) ;
      fprintf('Uploading %d movie files...\n', movieCount) ;
      % fileDescription = 'Movie file' ;
      % sidecarDescription = 'Movie sidecar file' ;
      for i = 1:movieCount ,
        wslPath = wslPathFromMovieIndex{i};
        remotePath = remotePathFromMovieIndex{i};
        %obj.uploadOrVerifySingleFile_(wslPath, remotePath, fileDescription) ;  % throws
        obj.rsyncUploadFile(wslPath, remotePath) ;  % throws
        % If there's a sidecar file, upload it too
        [~,~,fileExtension] = fileparts(wslPath) ;
        if strcmp(fileExtension,'.mjpg') ,
          sidecarWslPath = FSPath.replaceExtension(wslPath, '.txt') ;
          if exist(sidecarWslPath, 'file') ,
            sidecarRemotePath = obj.remote_path_from_wsl(sidecarWslPath) ;
            % obj.uploadOrVerifySingleFile_(sidecarWslPath, sidecarRemotePath, sidecarDescription) ;  % throws
            obj.rsyncUploadFile(sidecarWslPath, sidecarRemotePath) ;  % throws
          end
        end
      end      
      fprintf('Done uploading %d movie files.\n', movieCount) ;
      obj.didUploadMovies_ = true ; 
      obj.wslPathFromMovieIndex_ = wslPathFromMovieIndex ;
      obj.remotePathFromMovieIndex_ = remotePathFromMovieIndex ;
    end  % function
    
    % function uploadOrVerifySingleFile_(obj, localPath, remotePath, fileDescription)
    %   % Upload a single file.  Protected by convention.
    %   localFileDirOutput = dir(localPath) ;
    %   localFileSizeInKibibytes = round(localFileDirOutput.bytes/2^10) ;
    %   % We just use scpUploadOrVerify which does not confirm the identity
    %   % of file if it already exists. These movie files should be
    %   % immutable once created and their naming (underneath timestamped
    %   % modelchainIDs etc) should be pretty/totally unique. 
    %   %
    %   % Only situation that might cause problems are augmentedtrains but
    %   % let's not worry about that for now.
    %   localFileName = localFileDirOutput.name ;
    %   fullFileDescription = sprintf('%s (%s), %d KiB', fileDescription, localFileName, localFileSizeInKibibytes) ;
    %   obj.scpUploadOrVerify(localPath, ...
    %                         remotePath, ...
    %                         fullFileDescription, ...
    %                         'destRelative',false) ;  % throws      
    % end  % function
    
    % function result = getLocalMoviePathFromRemote(obj, queryRemotePath)
    %   if ~obj.didUploadMovies_ ,
    %     error('Can''t get a local movie path from a remote path if movies have not been uploaded.') ;
    %   end
    %   movieCount = numel(obj.remotePathFromMovieIndex_) ;
    %   for movieIndex = 1 : movieCount ,
    %     remotePath = obj.remotePathFromMovieIndex_{movieIndex} ;
    %     if strcmp(remotePath, queryRemotePath) ,
    %       result = obj.wslPathFromMovieIndex_{movieIndex} ;
    %       return
    %     end
    %   end
    %   % If we get here, queryRemotePath did not match any path in obj.remotePathFromMovieIndex_
    %   error('Query path %s does not match any remote movie path known to the backend.', queryRemotePath) ;
    % end  % function
    
    % function result = getRemoteMoviePathFromLocal(obj, queryWslPath)
    %   if ~obj.didUploadMovies_ ,
    %     error('Can''t get a remote movie path from a local path if movies have not been uploaded.') ;
    %   end
    %   movieCount = numel(obj.wslPathFromMovieIndex_) ;
    %   for movieIndex = 1 : movieCount ,
    %     wslPath = obj.wslPathFromMovieIndex_{movieIndex} ;
    %     if strcmp(wslPath, queryWslPath) ,
    %       result = obj.remotePathFromMovieIndex_{movieIndex} ;
    %       return
    %     end
    %   end
    %   % If we get here, queryLocalPath did not match any path in obj.localPathFromMovieIndex_
    %   error('Query path %s does not match any local movie path known to the backend.', queryWslPath) ;
    % end  % function
    
    function [isRunning, reasonNotRunning] = ensureIsRunning(obj)
      % If the AWS EC2 instance is not running, tell it to start, and wait for it to be
      % fully started.  On return, isRunning reflects whether this worked.  If
      % isRunning is false, reasonNotRunning is a string that says something about
      % what went wrong.

      % Make sure the credentials are set
      if ~obj.areCredentialsSet ,
        isRunning = false ;
        reasonNotRunning = 'AWS credentials are not set.' ;
        return          
      end
      
      % Make sure the instance ID is set
      if ~obj.isInstanceIDSet
        isRunning = false ;
        reasonNotRunning = 'AWS instance ID is not set.' ;
        return
      end

      % Make sure the instance exists
      [doesInstanceExist,isInstanceRunning] = obj.inspectInstance() ;
      if ~doesInstanceExist,
        isRunning = false;
        reasonNotRunning = sprintf('Instance %s could not be found.', obj.instanceID_) ;
        return
      end
      
      % Make sure the instance is running.  If not, start it.
      if ~isInstanceRunning ,
        % Instance is not running, so try to start it
        didStartInstance = obj.startInstance();
        if ~didStartInstance
          isRunning = false ;
          reasonNotRunning = sprintf('Could not start AWS EC2 instance %s.',obj.instanceID_) ;
          return
        end
      end
      
      % Just because you told EC2 to start the instance, and that worked, doesn't
      % mean the instance is truly ready.  Wait for it to be truly ready.
      isRunning = obj.waitForInstanceStart();
      if ~isRunning ,
        reasonNotRunning = 'Timed out waiting for AWS EC2 instance to be spooled up.';
        return
      end
      
      % If get here, all is well, EC2 instance is spun up and ready to go
      reasonNotRunning = '';
    end  % function    

    function updateRepo(obj)
      % Update the APT source code on the backend.  While we're at it, make sure the
      % pretrained weights are downloaded.
      obj.errorIfInstanceNotRunning();  % errs if instance isn't running

      % Does the APT source root dir exist?
      native_apt_root = APT.Root ;  % native path
      wsl_apt_root = wsl_path_from_native(native_apt_root) ;
      remote_apt_root = obj.remote_path_from_wsl(wsl_apt_root) ;  % remote path
      
      % Create folder if needed
      [didsucceed, msg] = obj.mkdir(wsl_apt_root) ;
      if ~didsucceed ,
        error('Unable to create APT source folder in AWS instance.\nStdout/stderr:\n%s\n', msg) ;
      end
      fprintf('APT source folder %s exists on AWS instance.\n', remote_apt_root);
      
      % Rsync the local APT code to the remote end
      obj.rsyncUploadFolder(wsl_apt_root, remote_apt_root) ;  % Will throw on error
      fprintf('Successfully rsynced remote APT source code (in %s) from local version (in %s).\n', remote_apt_root, native_apt_root) ;

      % Run the remote Python script to download the pretrained model weights
      % This python script doesn't do anything fancy, apparently, so we use the
      % python interpreter provided by the plain EC2 instance, not the one inside
      % the Docker container on the instance.
      download_script_path = linux_fullfile(remote_apt_root, 'deepnet', 'download_pretrained.py') ;
      quoted_download_script_path = escape_string_for_bash(download_script_path) ;      
      [st_3,res_3] = obj.runBatchCommandOutsideContainer(quoted_download_script_path) ;
      if st_3 ~= 0 ,
        error('Failed to download pretrained model weights:\n%s', res_3);
      end
      
      % If get here, all is well
      fprintf('Updated remote APT source code.\n\n');
    end  % function    
    
    function nframes = readTrkFileStatus(obj, wslFilePath, isTextFile, logger)
      % Read the number of frames remaining according to the remote file
      % corresponding to absolute local file path
      % localFilepath.  If partFileIsTextStatus is true, this file is assumed to be a
      % text file.  Otherwise, it is assumed to be a .mat file.  If the file does
      % not exist or there's some problem reading the file, returns nan.
      if ~exist('isTextFile', 'var') || isempty(isTextFile) ,
        isTextFile = false ;
      end
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger(1, 'DLBackEndClass::readTrkFileStatus()') ;
      end

      %logger.log('partFileIsTextStatus: %d', double(partFileIsTextStatus)) ;
      remoteFilePath = ...
         obj.remote_path_from_wsl(wslFilePath) ;
      if ~obj.fileExists(wslFilePath) ,
        nframes = nan ;
        return
      end
      if isTextFile,
        str = obj.fileContents(wslFilePath) ;
        nframes = TrkFile.getNFramesTrackedString(str) ;
      else
        nativeCopyFilePath = strcat(tempname(), '.mat') ;  % Has to have an extension or matfile() will add '.mat' to the filename
        wslCopyFilePath = wsl_path_from_native(nativeCopyFilePath) ;
        %logger.log('BgTrackWorkerObjAWS::readTrkFileStatus(): About to call obj.awsec2.scpDownloadOrVerify()...\n') ;
        % did_succeed = obj.scpDownloadOrVerify(remoteFilePath, localCopyFilePath) ;
        try
          obj.rsyncDownloadFile(remoteFilePath, wslCopyFilePath) ;  % throws on error
          %logger.log('Successfully downloaded remote tracking file %s\n', filename) ;
          nframes = TrkFile.getNFramesTrackedMatFile(nativeCopyFilePath) ;
          %logger.log('Read that nframes = %d\n', nframes) ;
        catch me
          logger.log('Could not download and/or read tracking progress from remote file %s:\n%s\n', remoteFilePath, me.getReport()) ;
          nframes = nan ;
        end
      end
    end  % function    

    function writeStringToFile(obj, fileWslPath, str)
      % Write the given string to a file, overrwriting any previous contents.
      % localFileAbsPath should be a WSL absolute path.
      % Throws if unable to write string to file.

      tfo = temp_file_object('w') ;  % local temp file, will be deleted when tfo goes out of scope
      tfo.fprintf('%s', str) ;
      tfo.fclose() ;  % Close the file before uploading to the remote side
      wsl_temp_file_path = wsl_path_from_native(tfo.abs_file_path) ;
      remoteFileAbsPath = obj.remote_path_from_wsl(fileWslPath) ;
      obj.rsyncUploadFile(wsl_temp_file_path, remoteFileAbsPath) ;
    end  % function
    
  end  % methods

  % These next two methods allow access to private and protected variables,
  % intended to be used for encoding/decoding.  The trailing underscore is there
  % to remind you that these methods are only intended for "special case" uses.
  methods
    function result = get_property_value_(self, name)
      result = self.(name) ;
    end  % function
    
    function set_property_value_(self, name, value)
      self.(name) = value ;
    end  % function
  end
  
  methods
    function result = remote_path_from_wsl(obj, wsl_path_or_paths)  % const method
      % Apply the applicable file name substitutions to path_or_paths.
      % path_or_paths can be a single path or a cellstring of paths, but all should
      % be WSL paths.
      %
      % This method does not mutate obj.
      
      if iscell(wsl_path_or_paths) ,
        wsl_paths = wsl_path_or_paths ;
        result = cellfun(@(wsl_path)(obj.applyFilePathSubstitutions(wsl_path)), wsl_paths) ;
      else
        wsl_path = wsl_path_or_paths ;
        result = obj.applyFilePathSubstitutions(wsl_path) ;
      end
    end  % function
    
    function result = applyFilePathSubstitutions(obj, command)  % const method
      % Apply the applicable file name substitutions to command.
      % This just uses dumb string replacement, which will likely lead to tears
      % eventually.  But to do better we'd have to keep commands as unescaped lists
      % until the last moment, and likely label which elements are paths.  This would be
      % great, but a lot of work.  Will put that off for now.
      %
      % This method does not mutate obj.
      
      result = ...
          AWSec2.applyGenericFilePathSubstitutions(command, ...
                                                   obj.wslProjectCachePath_, ...
                                                   obj.wslPathFromMovieIndex_) ;
    end  % function
  end  % methods
  
  methods (Static)
    function result = applyGenericFilePathSubstitutions(command, ...
                                                        wslProjectCachePath, ...
                                                        wslPathFromMovieIndex)
      % Deal with optional args
      if ~exist('wslPathFromMovieIndex', 'var') || isempty(wslPathFromMovieIndex) ,
        wslPathFromMovieIndex = cell(1,0) ;
      end
      % Replace the local cache root with the remote one
      if isempty(wslProjectCachePath)
        result_1 = command ;
      else
        result_1 = strrep(command, wslProjectCachePath, AWSec2.remoteDLCacheDir) ;
      end
      % Replace the local torch home path with the remote one
      result_1p5 = strrep(result_1, APT.gettorchhomepath(), AWSec2.remoteTorchHomeDir) ;      
      % Replace local movie paths with the corresponding remote ones
      if ~isempty(wslPathFromMovieIndex) ,
        remotePathFromMovieIndex = AWSec2.remote_movie_path_from_wsl(wslPathFromMovieIndex) ;
        result_2 = strrep_multiple(result_1p5, wslPathFromMovieIndex, remotePathFromMovieIndex) ;
      else
        result_2 = result_1p5 ;
      end
      % Replace the local APT source root with the remote one
      remote_apt_root = AWSec2.remoteAPTSourceRootDir ;
      wsl_apt_root = wsl_path_from_native(APT.Root) ;
      result_3 = strrep(result_2, wsl_apt_root, remote_apt_root) ;
      % Replace the local home dir with the remote one
      % Do this last b/c e.g. the local APT source root is likely in the local home
      % dir.
      wsl_home_path = wsl_path_from_native(get_home_dir_name()) ;
      remote_home_path = AWSec2.remoteHomeDir ;
      result_4 = strrep(result_3, wsl_home_path, remote_home_path) ;      
      result = result_4 ;
    end  % function

    function result = remote_movie_path_from_wsl(wsl_path_or_paths)
      % Convert a cell array of WSL movie paths to their remote equivalents.
      if iscell(wsl_path_or_paths)
        wsl_paths = wsl_path_or_paths ;
        result = cellfun(@(wsl_path)(AWSec2.single_remote_movie_path_from_wsl(wsl_path)), wsl_paths, 'UniformOutput', false) ;
      else
        % Assume its an old-school string
        wsl_path = wsl_path_or_paths ;
        result = AWSec2.single_remote_movie_path_from_wsl(wsl_path) ;
      end
    end

    function result = single_remote_movie_path_from_wsl(wsl_path)
      % Convert a single WSL movie path to the remote equivalent.
      movieName = fileparts23(wsl_path) ;
      rawRemotePath = linux_fullfile(AWSec2.remoteMovieCacheDir, movieName) ;
      result = FSPath.standardPath(rawRemotePath);  % transform to standardized linux-style path
    end
    
  end  % methods (Static)
end  % classdef
