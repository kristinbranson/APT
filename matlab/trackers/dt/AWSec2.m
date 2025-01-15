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
    AMI = 'ami-09b1db2d5c1d91c38';  % Deep Learning Base *Proprietary* Nvidia Driver GPU AMI (Ubuntu 20.04) 20240415, with conda
                                    % and apt_20230427_tf211_pytorch113_ampere environment, and ~/APT, and python
                                    % links in ~/bin, and dotfiles setup to setup the the path properly for ssh
                                    % noninteractive shells.  This was originally based on the image
                                    % ami-09b1db2d5c1d91c38, aka "Deep Learning Base Proprietary Nvidia Driver GPU
                                    % AMI (Ubuntu 20.04) 20240101"    
    autoShutdownAlarmNamePat = 'aptAutoShutdown'; 
    remoteHomeDir = '/home/ubuntu'
    remoteDLCacheDir = linux_fullfile(AWSec2.remoteHomeDir, 'cacheDL')
    remoteMovieCacheDir = linux_fullfile(AWSec2.remoteHomeDir, 'movies')
    remoteAPTSourceRootDir = linux_fullfile(AWSec2.remoteHomeDir, 'APT')
    scpCmd = AWSec2.computeScpCmd()
    rsyncCmd = AWSec2.computeRsyncCmd()
  end
  
  properties
    instanceID = ''  % Durable identifier for the AWS EC2 instance.  E.g.'i-07a3a8281784d4a38'.
  end

  properties (Dependent)
    isInstanceIDSet    % Whether the instanceID is set or not
    areCredentialsSet  % Whether the security credentials for the instance are set
    isInDebugMode      % Whether the object is in debug mode.  See isInDebugMode_
  end

  properties
    keyName = ''  % key(pair) name used to authenticate to AWS EC2, e.g. 'alt_taylora-ws4'
    pem = ''  % path to .pem file that holds an RSA private key used to ssh into the AWS EC2 instance
    instanceType = 'p3.2xlarge'  % the AWS EC2 machine instance type to use
  end

  properties (Transient, SetAccess=protected)
    % The backend keeps track of whether the DMCoD is local or remote.  When it's
    % remote, we substitute the remote DMC root for the local one wherever it
    % appears.
    isDMCRemote_ = false
      % True iff the "current" version of the DMC is on a remote AWS filesystem.  
      % Underscore means "protected by convention"    
    localDMCRootDir_ = '' ;  % e.g. /groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5
    %remoteDMCRootDir_ = '' ;  % e.g. /home/ubuntu/cacheDL    

    % Used to keep track of whether movies have been uploaded or not.
    % Transient and protected in spirit.
    didUploadMovies_ = false

    % When we upload movies, keep track of the correspondence, so we can help the
    % consumer map between the paths.  Transient, protected in spirit.
    localPathFromMovieIndex_ = cell(1,0) ;
    remotePathFromMovieIndex_ = cell(1,0) ;
  end

  properties (Dependent)
    isDMCRemote
    isDMCLocal    
    localDMCRootDir
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
    function set.instanceID(obj,v)
      obj.instanceID = v;
    end

    function v = get.areCredentialsSet(obj)
      v = ~isempty(obj.pem) && ~isempty(obj.keyName);
    end

    function v = get.isInstanceIDSet(obj)
      v = ~isempty(obj.instanceID);
    end

    function result = get.isInDebugMode(obj)
      result = obj.isInDebugMode_ ;
    end

    function set.isInDebugMode(obj, value)
      obj.isInDebugMode_ = value ;
    end

    function setInstanceIDAndType(obj,instanceID,instanceType)
      %obj.SetStatus(sprintf('Setting AWS EC2 instance = %s',instanceID));
      if ~isempty(obj.instanceID),
        if strcmp(instanceID,obj.instanceID),
          % nothing to do
          %obj.ClearStatus();
          return;
        end
        %instanceID = obj.instanceID;
        [tfexist,tfrunning] = obj.inspectInstance();
        if tfexist && tfrunning,
          tfsucc = obj.stopInstance();
          if ~tfsucc,
            warning('Error stopping old AWS EC2 instance %s.',instanceID);
          end
          %obj.SetStatus(sprintf('Setting AWS EC2 instance = %s',instanceID));
        end
      end
      obj.instanceID = instanceID;
      if nargin > 3,
        obj.instanceType = instanceType;
      end
      if obj.isInstanceIDSet ,
        obj.configureAlarm();
      end
      %obj.ClearStatus();
    end

    function [tfsucc,json] = launchInstance(obj,varargin)
      % Launch a brand-new instance to specify an unspecified instance
      [dryrun,dostore] = myparse(varargin,'dryrun',false,'dostore',true);
      obj.clearInstanceID();
      %obj.SetStatus('Launching new AWS EC2 instance');
      cmd = AWSec2.launchInstanceCmd(obj.keyName,'instType',obj.instanceType,'dryrun',dryrun);
      [st,json] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (st==0) ;
      if ~tfsucc
        %obj.ClearStatus();
        return;
      end
      json = jsondecode(json);
      instanceID = json.Instances.InstanceId;
      if dostore,
        obj.setInstanceIDAndType(instanceID);
      end
      %obj.SetStatus('Waiting for AWS EC2 instance to spool up.');
      [tfsucc] = obj.waitForInstanceStart();
      if ~tfsucc,
        %obj.ClearStatus();
        return;
      end
      obj.configureAlarm();
      %obj.ClearStatus();
    end
    
    function [tfexist,tfrunning,json] = inspectInstance(obj,varargin)
      % Check that a specified instance exists; check if it is running; 
      % get json; sets .instanceIP if running
      % 
      % * tfexist is returned as true of the instance exists in some state.
      % * tfrunning is returned as true if the instance exists and is running.
      % * json is valid only if tfexist==true.
      
      assert(obj.isInstanceIDSet,'Cannot inspect an unspecified AWSEc2 instance.');
      
      % Aside: this works with empty .instanceID if there is only one 
      % instance in the cloud, but we are not interested in that for now
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); 
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
      assert(strcmp(obj.instanceID,inst.InstanceId));
      if strcmpi(inst.State.Name,'terminated'),
        tfexist = false;
        tfrunning = false;
        return;
      end
      
      tfrunning = strcmp(inst.State.Name,'running');
      if tfrunning
        obj.instanceIP = inst.PublicIpAddress;
        fprintf('EC2 instanceID %s is running with IP %s.\n',...
          obj.instanceID,obj.instanceIP);
      else
        % leave IP for now even though may be outdated
      end
    end  % function
    
    function [tfsucc,state,json] = getInstanceState(obj)
      assert(obj.isInstanceIDSet);
      state = '';
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
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
      fprintf('Stopping AWS EC2 instance %s...\n',obj.instanceID);
      cmd = AWSec2.stopInstanceCmd(obj.instanceID);
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
      %obj.SetStatus(sprintf('Starting instance %s',obj.instanceID));
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
        cmd = AWSec2.startInstanceCmd(obj.instanceID);
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
        error('EC2 instance with id %s does not seem to exist', obj.instanceID);
      end
      if ~isInstanceRunning
        error('EC2 instance with id %s is not in the ''running'' state.',...
              obj.instanceID)
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
      
      instID = obj.instanceID;
      namestr = sprintf(obj.autoShutdownAlarmNamePat);
      descstr = sprintf('"Auto shutdown %d sec (%d periods)"',periodsec,evalperiods);
      dimstr = sprintf('"Name=InstanceId,Value=%s"',instID);
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
          if ~isempty(j) && strcmp(json.MetricAlarms(i).Dimensions(j).Value,obj.instanceID),
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
    
    function tfsucc = scpDownloadOrVerify(obj,srcAbs,dstAbs,varargin)
      % If dstAbs already exists, does NOT check identity of file against
      % dstAbs. In many cases, naming/immutability of files (with paths)
      % means this is OK.
      
      [sysCmdArgs] = ...
        myparse(varargin,...
                'sysCmdArgs',{}) ;
      
      if exist(dstAbs,'file') ,
        fprintf('File %s exists, not downloading.\n',dstAbs);
        tfsucc = true;
      else
        %logger.log('AWSSec2::scpDownloadOrVerify(): obj.scpcmd is %s\n', obj.scpCmd) ;
        cmd = AWSec2.scpDownloadCmd(obj.pem, obj.instanceIP, srcAbs, dstAbs, ...
                                    'scpcmd', obj.scpCmd) ;
        %logger.log('AWSSec2::scpDownloadOrVerify(): cmd is %s\n', cmd) ;
        st = AWSec2.syscmd(cmd,sysCmdArgs{:});
        tfsucc = (st==0) ;        
        tfsucc = tfsucc && (exist(dstAbs,'file')>0);
      end
    end
    
    function tfsucc = scpDownloadOrVerifyEnsureDir(obj,srcAbs,dstAbs,varargin)
      dirLcl = fileparts(dstAbs);
      if exist(dirLcl,'dir')==0
        [tfsucc,msg] = mkdir(dirLcl);
        if ~tfsucc
          warningNoTrace('Failed to create local directory %s: %s',dirLcl,msg);
          return;
        end
      end
      tfsucc = obj.scpDownloadOrVerify(srcAbs,dstAbs,varargin{:});
    end
 
    function tfsucc = scpUpload(obj,file,dest,varargin)
      [destRelative,sysCmdArgs] = myparse(varargin,...
        'destRelative',true,... % true if dest is relative to ~
        'sysCmdArgs',{});
      cmd = AWSec2.scpPrepareUploadCmd(obj.pem,obj.instanceIP,dest,...
                                       'destRelative',destRelative);
      AWSec2.syscmd(cmd,sysCmdArgs{:});
      cmd = AWSec2.scpUploadCmd(file,obj.pem,obj.instanceIP,dest,...
                                'scpcmd',obj.scpCmd,'destRelative',destRelative);
      st = AWSec2.syscmd(cmd,sysCmdArgs{:});
      tfsucc = (st==0) ;
    end
    
    function scpUploadOrVerify(obj,src,dst,fileDescStr,varargin) % throws
      % Either i) confirm a remote file exists, or ii) upload it.
      % In the case of i), NO CHECK IS MADE that the existing file matches
      % the local file.
      %
      % Could use rsync here instead but rsync is less likely to be 
      % installed on a Win machine
      %
      % This method either succeeds or fails and harderrors.
      %
      % src: full path to local file
      % dstRel: relative (to home) path to destination
      % fileDescStr: eg 'training file' or 'movie'
            
      destRelative = myparse(varargin,...
        'destRelative',true);
      
      if destRelative
        dstAbs = ['~/' dst];
      else
        dstAbs = dst;
      end
      
      src_d = dir(src);
      src_sz = src_d.bytes;
      tfsucc = obj.fileExistsAndIsGivenSize(dstAbs,src_sz);
      if tfsucc
        fprintf('%s file exists: %s.\n\n',...
          String.niceUpperCase(fileDescStr),dstAbs);
      else
        %obj.SetStatus(sprintf('Uploading %s file to AWS EC2 instance',fileDescStr));
        fprintf('About to upload. This could take a while depending ...\n');
        tfsucc = obj.scpUpload(src,dstAbs,...
                               'destRelative',false,'sysCmdArgs',{});
        %obj.ClearStatus();
        if tfsucc
          fprintf('Uploaded %s %s to %s.\n\n',fileDescStr,src,dst);
        else
          error('Failed to upload %s %s.',fileDescStr,src);
        end
      end
    end
    
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
       
    function tfsucc = rsyncUpload(obj, src, dest)
      cmd = AWSec2.rsyncUploadCmd(src, obj.pem, obj.instanceIP, dest) ;
      st = AWSec2.syscmd(cmd) ;
      tfsucc = (st==0) ;
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
    
    function tf = fileExists(obj,f)
      %script = '/home/ubuntu/APT/matlab/misc/fileexists.sh';
      %cmdremote = sprintf('%s %s',script,f);
      cmdremote = sprintf('/usr/bin/test -e %s ; echo $?',f);
      [~,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err'); 
      tf = strcmp(strtrim(res),'0') ;      
    end
    
    function tf = fileExistsAndIsNonempty(obj,f)
      %script = '/home/ubuntu/APT/matlab/misc/fileexistsnonempty.sh';
      %cmdremote = sprintf('%s %s',script,f);
      cmdremote = sprintf('/usr/bin/test -s %s ; echo $?',f);
      [~,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err'); 
      tf = strcmp(strtrim(res),'0') ;      
    end
    
    function tf = fileExistsAndIsGivenSize(obj,f,query_byte_count)
      %script = '/home/ubuntu/APT/matlab/misc/fileexists.sh';
      %cmdremote = sprintf('%s %s %d',script,f,size);
      cmdremote = sprintf('/usr/bin/stat --printf="%%s\\n" %s',f) ;
      [~,res] = obj.runBatchCommandOutsideContainer(cmdremote,'failbehavior','err'); 
      actual_byte_count = str2double(res) ;
      tf = (actual_byte_count == query_byte_count) ;      
    end
    
    function s = fileContents(obj,f,varargin)
      % First check if the file exists
      cmdremote = sprintf('/usr/bin/test -e %s ; echo $?',f);
      [st,res] = obj.runBatchCommandOutsideContainer(cmdremote, 'failbehavior', 'silent', varargin{:}) ;
      if st==0 ,
        % Command succeeded in determining whether the file exists
        if strcmp(strtrim(res),'0') ,
          % File exists
          cmdremote = sprintf('cat %s',f);
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
    
    function result = remoteFileModTime(obj, filename, varargin)
      % Returns the file modification time (mtime) in seconds since Epoch
      command = sprintf('stat --format=%%Y %s', escape_string_for_bash(filename)) ;  % time of last data modification, seconds since Epoch
      [st, stdouterr] = obj.runBatchCommandOutsideContainer(command, varargin{:}) ; 
      did_succeed = (st==0) ;
      if did_succeed ,
        result = str2double(stdouterr) ;
      else
        % Warning/error happens inside obj.runBatchCommandOutsideContainer(), so just set a fallback value
        result = nan ;
      end
    end
    
    function tfsucc = lsdir(obj,remoteDir,varargin)
      [failbehavior,args] = myparse(varargin,...
                                    'failbehavior','warn',...
                                    'args','-lha') ;      
      cmdremote = sprintf('ls %s %s',args,remoteDir);
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
    
    function remotePaths = remoteGlob(obj,globs)
      % Look for remote files/paths. Either succeeds, or fails and harderrors.

      % globs: cellstr of globs
      
      lscmd = cellfun(@(glob)sprintf('ls %s 2> /dev/null ; ',glob), globs, 'uni', 0) ;
      lscmd = cat(2,lscmd{:});
      [st,res] = obj.runBatchCommandOutsideContainer(lscmd);
      tfsucc = (st==0) ;      
      if tfsucc
        remotePaths = regexp(res,'\n','split');
        remotePaths = remotePaths(:);
        remotePaths = remotePaths(~cellfun(@isempty,remotePaths));
      else
        error('Failed to find remote files/paths %s: %s',...
          String.cellstr2CommaSepList(globs),res);
      end      
    end
    
    function result = wrapCommandSSH(obj, input_command, varargin)
      remote_command_with_file_name_substitutions = ...
        AWSec2.applyFileNameSubstitutions(input_command, ...
                                          obj.isDMCRemote_, obj.localDMCRootDir_, ...
                                          obj.didUploadMovies_, obj.localPathFromMovieIndex_, obj.remotePathFromMovieIndex_) ;
      result = wrapCommandSSH(remote_command_with_file_name_substitutions, ...
                              'host', obj.instanceIP, ...
                              'timeout',8, ...
                              'username', 'ubuntu', ...
                              'identity', obj.pem) ;
    end  % function

    function [st,res] = runBatchCommandOutsideContainer(obj, cmdremote, varargin)      
      % Runs a single command-line command on the ec2 instance.

      % Wrap for ssh'ing into an AWS instance
      cmd1 = obj.wrapCommandSSH(cmdremote) ;  % uses fields of obj to set parameters for ssh command
    
      % Need to prepend a sleep to avoid problems
      precommand = 'sleep 5 && export AWS_PAGER=' ;
        % Change the sleep value at your peril!  I changed it to 3 and everything
        % seemed fine for a while, until it became a very hard-to-find bug!  
        % --ALT, 2024-09-12
      command = sprintf('%s && %s', precommand, cmd1) ;

      % Issue the command, gather results
      [st, res] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, varargin{:}) ;      
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

    function clearInstanceID(obj)
      obj.setInstanceIDAndType('');
    end
    
%     function tf = isSameInstance(obj,obj2)
%       assert(isscalar(obj) && isscalar(obj2));
%       tf = strcmp(obj.instanceID,obj2.instanceID) && ~isempty(obj.instanceID);
%     end
  end  % methods
  
  methods (Static)
    
    function cmd = launchInstanceCmd(keyName,varargin)
      [ami,instType,secGrp,dryrun] = myparse(varargin,...
        'ami',AWSec2.AMI,...
        'instType','p3.2xlarge',...
        'secGrp',AWSec2.AWS_SECURITY_GROUP,...
        'dryrun',false);
      cmd = sprintf('aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s',ami,instType,secGrp);
      if dryrun,
        cmd = [cmd,' --dry-run'];
      end
      if ~isempty(keyName),
        cmd = [cmd,' --key-name ',keyName];
      end
    end
    
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

    function cmd = rsyncUploadCmd(src, pemFilePath, ip, dest)
      % Generate the system() command to upload a file/folder via rsync.

      % It's important that neither src nor dest have a trailing slash
      if isempty(src) ,
        error('src folder for rsync cannot be empty') ;
      else
        if strcmp(src(end),'/') ,
          error('src folder for rsync cannot end in a slash') ;
        end
      end
      if isempty(dest) ,
        error('dest folder for rsync cannot be empty') ;
      else
        if strcmp(dest(end),'/') ,
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
      cmd = sprintf('%s --rsh=%s %s/ ubuntu@%s:%s', AWSec2.rsyncCmd, escaped_sshcmd, src, ip, dest) ;
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

    function scpCmd = computeScpCmd()
      if ispc()
        windows_null_device_path = '\\.\NUL' ;
        scpCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSCPCMD, windows_null_device_path) ; 
      else
        scpCmd = 'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
      end
    end

%     function sshCmd = computeSshCmd()
%       if ispc()
%         windows_null_device_path = '\\.\NUL' ;
%         sshCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSSHCMD, windows_null_device_path) ; 
%       else
%         sshCmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
%       end
%     end
    
    function result = computeRsyncCmd()
      if ispc()
        error('Not implemented') ;
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
      suitcase.isDMCRemote_ = obj.isDMCRemote_ ;
      suitcase.localDMCRootDir_ = obj.localDMCRootDir_ ;
      suitcase.didUploadMovies_ = obj.didUploadMovies_ ;
      suitcase.localPathFromMovieIndex_ = obj.localPathFromMovieIndex_ ;
      suitcase.remotePathFromMovieIndex_ = obj.remotePathFromMovieIndex_ ;
    end  % function
    
    function restoreAfterParfeval(obj, suitcase)
      % Should be called in background tasks run via parfeval, to restore fields that
      % should not be restored from persistence, but we want to survive the parfeval
      % boundary.
      obj.instanceIP = suitcase.instanceIP ;
      obj.isInDebugMode_ = suitcase.isInDebugMode_ ;
      obj.isDMCRemote_ = suitcase.isDMCRemote_ ;
      obj.localDMCRootDir_ = suitcase.localDMCRootDir_ ;
      obj.didUploadMovies_ = suitcase.didUploadMovies_ ;
      obj.localPathFromMovieIndex_ = suitcase.localPathFromMovieIndex_ ;
      obj.remotePathFromMovieIndex_ = suitcase.remotePathFromMovieIndex_ ;
    end  % function

    function [isAllWell, message] = downloadTrackingFilesIfNecessary(obj, res, localCacheRoot, movfiles)
      remoteCacheRoot = AWSec2.remoteDLCacheDir ;
      currentLocalPathFromTrackedMovieIndex = movfiles(:) ;  % column cellstr
      originalLocalPathFromTrackedMovieIndex = {res.movfile}' ;  % column cellstr
      if all(strcmp(currentLocalPathFromTrackedMovieIndex,originalLocalPathFromTrackedMovieIndex))
        % we perform this check b/c while tracking has been running in
        % the bg, the project could have been updated, movies
        % renamed/reordered etc.        
        % download trkfiles 
        localTrackFilePaths = {res.trkfile} ;
        remoteTrackFilePaths = replace_prefix_path(localTrackFilePaths, localCacheRoot, remoteCacheRoot) ;
        sysCmdArgs = {'failbehavior', 'err'};
        for ivw=1:numel(res)
          trkRmt = remoteTrackFilePaths{ivw};
          trkLcl = localTrackFilePaths{ivw};
          fprintf('Trying to download %s to %s...\n',trkRmt,trkLcl);
          obj.scpDownloadOrVerifyEnsureDir(trkRmt,trkLcl,'sysCmdArgs',sysCmdArgs); % XXX doc orVerify
          fprintf('Done downloading %s to %s...\n',trkRmt,trkLcl);
        end
        isAllWell = true ;
        message = '' ;
      else
        isAllWell = false ;
        message = sprintf('Tracking complete, but one or move movies has been changed in current project.') ;
        % conservative, take no action for now
        return
      end
    end  % function    

    function result = get.isDMCRemote(obj)
      result = obj.isDMCRemote_ ;
    end  % function

    function result = get.isDMCLocal(obj)
      result = ~obj.isDMCRemote_ ;
    end  % function    

    function [tfsucc,res] = batchPoll(obj, fspollargs)
      % fspollargs: [n] cellstr eg {'exists' '/my/file' 'existsNE' '/my/file2'}
      %
      % res: [n] cellstr of fspoll responses

      assert(iscellstr(fspollargs) && ~isempty(fspollargs));  %#ok<ISCLSTR> 
      nargsFSP = numel(fspollargs);
      assert(mod(nargsFSP,2)==0);
      nresps = nargsFSP/2;
      
      fspollstr = space_out(fspollargs);
      fspoll_script_path = '/home/ubuntu/APT/matlab/misc/fspoll.py' ;

      cmdremote = sprintf('%s %s',fspoll_script_path,fspollstr);

      [st,res] = obj.runBatchCommandOutsideContainer(cmdremote);
      tfsucc = (st==0) ;
      if tfsucc
        res = regexp(res,'\n','split');
        tfsucc = iscell(res) && numel(res)==nresps+1; % last cell is {0x0 char}
        res = res(1:end-1);
      else
        res = [];
      end
    end  % function
    
    function maxiter = getMostRecentModel(obj, dmc)  % constant method
      if obj.isDMCRemote_ ,
        % maxiter is nan if something bad happened or if DNE
        % TODO allow polling for multiple models at once
        [dirModelChainLnx,idx] = dmc.dirModelChainLnx();
        fspollargs = {};
        for i = 1:numel(idx),
          fspollargs = [fspollargs,{'mostrecentmodel' dirModelChainLnx{i}}]; %#ok<AGROW>
        end
        [tfsucc,res] = obj.batchPoll(fspollargs);
        if tfsucc
          maxiter = str2double(res(1:numel(idx))); % includes 'DNE'->nan
        else
          maxiter = nan(1,numel(idx));
        end        
      else
        maxiter = dmc.getMostRecentModelLocal() ;
      end
    end  % function
    
    function [didsucceed, msg] = mkdir(obj, dir_name)
      % Create the named directory, either locally or remotely, depending on the
      % backend type.      
      quoted_dirloc = escape_string_for_bash(dir_name) ;
      base_command = sprintf('mkdir -p %s', quoted_dirloc) ;
      [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;
      didsucceed = (status==0) ;
    end
    
    function mirrorDMCToBackend(obj, dmc, mode)
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
      assert(isa(dmc, 'DeepModelChainOnDisk')) ;      
      assert(isscalar(dmc));

      % If the DMC is already remote, warn
      if obj.isDMCRemote ,
        warning('Mirroring DMC to backend, even though DMC is already remote.') ;
      end

      % Make sure there is a trained model
      maxiter = obj.getMostRecentModel(dmc) ;
      succ = (maxiter >= 0) ;
      if strcmp(mode, 'tracking') && any(~succ) ,
        dmclfail = dmc.dirModelChainLnx(find(~succ));
        fstr = sprintf('%s ',dmclfail{:});
        error('Failed to determine latest model iteration in %s.',fstr);
      end
      if isnan(maxiter) ,
        fprintf('Currently, there is no trained model.\n');
      else
        fprintf('Current model iteration is %s.\n',mat2str(maxiter));
      end
     
      % Make sure there is a live backend
      obj.errorIfInstanceNotRunning();  % throws error if ec2 instance is not connected
      
      % To support training on AWS, and the fact that a DeepModelChainOnDisk has
      % only a single boolean to represent whether it's local or remote, we're just
      % going to upload everything under fullfile(obj.rootDir, obj.projID) to the
      % backend.  -- ALT, 2024-06-25
      localProjectPath = fullfile(dmc.rootDir, dmc.projID) ;
      remoteProjectPath = linux_fullfile(AWSec2.remoteDLCacheDir, dmc.projID) ;  % ensure linux-style path
      [didsucceed, msg] = obj.mkdir(remoteProjectPath) ;
      if ~didsucceed ,
        error('Unable to create remote dir %s.\nmsg:\n%s\n', remoteProjectPath, msg) ;
      end
      obj.rsyncUpload(localProjectPath, remoteProjectPath) ;

      % If we made it here, upload successful---update the state to reflect that the
      % model is now remote.      
      %obj.remoteDMCRootDir_ = AWSec2.remoteDLCacheDir ;
      obj.localDMCRootDir_ = dmc.rootDir ;
      obj.isDMCRemote_ = true ;
    end  % function
    
    function mirrorDMCFromBackend(obj, dmc)
      % Inverse of mirror2remoteAws. Download/mirror model from remote AWS
      % instance to local cache.
      %
      % update .rootDir, .reader appropriately to point to model in local
      % cache.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the local cache.
      
      assert(isa(dmc, 'DeepModelChainOnDisk')) ;      
      assert(isscalar(dmc));      

      % If the DMC is already local, warn
      if obj.isDMCLocal ,
        warning('Mirroring DMC from backend, even though DMC is already local.') ;
      end
 
      maxiter = obj.getMostRecentModel(dmc) ;
      succ = (maxiter >= 0) ;
      if any(~succ),
        dirModelChainLnx = dmc.dirModelChainLnx(find(~succ));
        fstr = sprintf('%s ',dirModelChainLnx{:});
        error('Failed to determine latest model iteration in %s.',...
          fstr);
      end
      fprintf('Current model iteration is %s.\n',mat2str(maxiter));
     
      [tfexist,tfrunning] = obj.inspectInstance();
      if ~tfexist,
        error('AWS EC2 instance %s could not be found.',obj.instanceID);
      end
      if ~tfrunning,
        [tfsucc,~,warningstr] = obj.startInstance();
        if ~tfsucc,
          error('Could not start AWS EC2 instance %s: %s',obj.instanceID,warningstr);
        end
      end      
          
      localDMCRootDir = obj.localDMCRootDir_ ;
      modelGlobsLnx = dmc.modelGlobsLnx();
      n = dmc.n ;
      remoteDMCRootDir = AWSec2.remoteDLCacheDir ;
      dmcNetType = dmc.netType ;
      for j = 1:n,
        mdlFilesRemote = obj.remoteGlob(modelGlobsLnx{j});
        cacheDirLocalEscd = regexprep(localDMCRootDir,'\\','\\\\');
        mdlFilesLcl = regexprep(mdlFilesRemote,remoteDMCRootDir,cacheDirLocalEscd);
        nMdlFiles = numel(mdlFilesRemote);
        netstr = char(dmcNetType{j}); 
        fprintf(1,'Download/mirror %d model files for net %s.\n',nMdlFiles,netstr);
        for i=1:nMdlFiles
          fsrc = mdlFilesRemote{i};
          fdst = mdlFilesLcl{i};
          % See comment in mirror2RemoteAws regarding not confirming ID of
          % files-that-already-exist
          obj.scpDownloadOrVerifyEnsureDir(fsrc,fdst,...
            'sysCmdArgs',{'failbehavior', 'err'}); % throws
        end
      end      
      % if we made it here, download successful
      
      %obj.rootDir = cacheDirLocal;
      %obj.reader = DeepModelChainReaderLocal();
      obj.isDMCRemote_ = false ;
    end  % function
        
    function result = getTorchHome(obj)
      if obj.isDMCRemote_ ,
        result = linux_fullfile(AWSec2.remoteDLCacheDir, 'torch') ;
      else
        result = fullfile(APT.getdotaptdirpath(), 'torch') ;
      end
    end  % function
    
    function result = get.localDMCRootDir(obj) 
      result = obj.localDMCRootDir_ ;
    end  % function

    function set.localDMCRootDir(obj, value) 
      obj.localDMCRootDir_ = value ;
    end  % function
    
    % function result = get.remoteDMCRootDir(obj)
    %   result = AWSec2.remoteDLCacheDir ;
    % end  % function
        
    function uploadMovies(obj, localPathFromMovieIndex)
      % Upload movies to the backend, if necessary.
      if obj.didUploadMovies_ ,
        return
      end
      remotePathFromMovieIndex = AWSec2.remoteMoviePathsFromLocal(localPathFromMovieIndex) ;
      movieCount = numel(localPathFromMovieIndex) ;
      fprintf('Uploading %d movie files...\n', movieCount) ;
      fileDescription = 'Movie file' ;
      sidecarDescription = 'Movie sidecar file' ;
      for i = 1:movieCount ,
        localPath = localPathFromMovieIndex{i};
        remotePath = remotePathFromMovieIndex{i};
        obj.uploadOrVerifySingleFile_(localPath, remotePath, fileDescription) ;  % throws
        % If there's a sidecar file, upload it too
        [~,~,fileExtension] = fileparts(localPath) ;
        if strcmp(fileExtension,'.mjpg') ,
          sidecarLocalPath = FSPath.replaceExtension(localPath, '.txt') ;
          if exist(sidecarLocalPath, 'file') ,
            sidecarRemotePath = AWSec2.remoteMoviePathFromLocal(sidecarLocalPath) ;
            obj.uploadOrVerifySingleFile_(sidecarLocalPath, sidecarRemotePath, sidecarDescription) ;  % throws
          end
        end
      end      
      fprintf('Done uploading %d movie files.\n', movieCount) ;
      obj.didUploadMovies_ = true ; 
      obj.localPathFromMovieIndex_ = localPathFromMovieIndex ;
      obj.remotePathFromMovieIndex_ = remotePathFromMovieIndex ;
    end  % function
    
    function uploadOrVerifySingleFile_(obj, localPath, remotePath, fileDescription)
      % Upload a single file.  Protected by convention.
      localFileDirOutput = dir(localPath) ;
      localFileSizeInKibibytes = round(localFileDirOutput.bytes/2^10) ;
      % We just use scpUploadOrVerify which does not confirm the identity
      % of file if it already exists. These movie files should be
      % immutable once created and their naming (underneath timestamped
      % modelchainIDs etc) should be pretty/totally unique. 
      %
      % Only situation that might cause problems are augmentedtrains but
      % let's not worry about that for now.
      localFileName = localFileDirOutput.name ;
      fullFileDescription = sprintf('%s (%s), %d KiB', fileDescription, localFileName, localFileSizeInKibibytes) ;
      obj.scpUploadOrVerify(localPath, ...
                            remotePath, ...
                            fullFileDescription, ...
                            'destRelative',false) ;  % throws      
    end  % function
    
    function result = getLocalMoviePathFromRemote(obj, queryRemotePath)
      if ~obj.didUploadMovies_ ,
        error('Can''t get a local movie path from a remote path if movies have not been uploaded.') ;
      end
      movieCount = numel(obj.remotePathFromMovieIndex_) ;
      for movieIndex = 1 : movieCount ,
        remotePath = obj.remotePathFromMovieIndex_{movieIndex} ;
        if strcmp(remotePath, queryRemotePath) ,
          result = obj.localPathFromMovieIndex_{movieIndex} ;
          return
        end
      end
      % If we get here, queryRemotePath did not match any path in obj.remotePathFromMovieIndex_
      error('Query path %s does not match any remote movie path known to the backend.', queryRemotePath) ;
    end  % function
    
    function result = getRemoteMoviePathFromLocal(obj, queryLocalPath)
      if ~obj.didUploadMovies_ ,
        error('Can''t get a remote movie path from a local path if movies have not been uploaded.') ;
      end
      movieCount = numel(obj.localPathFromMovieIndex_) ;
      for movieIndex = 1 : movieCount ,
        localPath = obj.localPathFromMovieIndex_{movieIndex} ;
        if strcmp(localPath, queryLocalPath) ,
          result = obj.remotePathFromMovieIndex_{movieIndex} ;
          return
        end
      end
      % If we get here, queryLocalPath did not match any path in obj.localPathFromMovieIndex_
      error('Query path %s does not match any local movie path known to the backend.', queryLocalPath) ;
    end  % function
    
    function [isRunning, reasonNotRunning] = ensureIsRunning(obj)
      % If the AWS EC2 instance is not running, tell it to start, and wait for it to be
      % fully started.  On return, isRunning reflects whether this worked.  If
      % isRunning is false, reasonNotRunning is a string that says something about
      % what went wrong.

      % Make sure the instance ID is set
      if ~obj.isInstanceIDSet
        isRunning = false ;
        reasonNotRunning = 'AWS instance ID is not set.' ;
        return
      end

      % Make sure the credentials are set
      if ~obj.areCredentialsSet ,
        isRunning = false ;
        reasonNotRunning = 'AWS credentials are not set.' ;
        return          
      end
      
      % Make sure the instance exists
      [doesInstanceExist,isInstanceRunning] = obj.inspectInstance() ;
      if ~doesInstanceExist,
        isRunning = false;
        reasonNotRunning = sprintf('Instance %s could not be found.', obj.instanceID) ;
        %obj.awsec2.clearInstanceID();  % Don't think we want to do this just yet
        return
      end
      
      % Make sure the instance is running.  If not, start it.
      if ~isInstanceRunning ,
        % Instance is not running, so try to start it
        didStartInstance = obj.startInstance();
        if ~didStartInstance
          isRunning = false ;
          reasonNotRunning = sprintf('Could not start AWS EC2 instance %s.',obj.instanceID) ;
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
      remote_apt_root = AWSec2.remoteAPTSourceRootDir ;
      
      % Create folder if needed
      [didsucceed, msg] = obj.mkdir(remote_apt_root) ;
      if ~didsucceed ,
        error('Unable to create APT source folder in AWS instance.\nStdout/stderr:\n%s\n', msg) ;
      end
      fprintf('APT source folder %s exists on AWS instance.\n', remote_apt_root);

      % Rsync the local APT code to the remote end
      local_apt_root = APT.Root ;
      tfsucc = obj.rsyncUpload(local_apt_root, remote_apt_root) ;
      if tfsucc ,
        fprintf('Successfully rsynced remote APT source code (in %s) with local version (in %s).\n', remote_apt_root, local_apt_root) ;
      else
        error('Unable to rsync remote APT source code (in %s) with local version (in %s)', remote_apt_root, local_apt_root) ;
      end

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
  
  methods (Static)
    function result = remoteMoviePathFromLocal(localPath)
      % Convert a local movie path to the remote equivalent.
      movieName = fileparts23(localPath) ;
      rawRemotePath = linux_fullfile(AWSec2.remoteMovieCacheDir, movieName) ;
      result = FSPath.standardPath(rawRemotePath);  % transform to standardized linux-style path
    end

    function result = remoteMoviePathsFromLocal(localPathFromMovieIndex)
      % Convert a cell array of local movie paths to their remote equivalents.
      % For non-AWS backends, this is the identity function.
      result = cellfun(@(path)(AWSec2.remoteMoviePathFromLocal(path)), localPathFromMovieIndex, 'UniformOutput', false) ;
    end

    function result = applyFileNameSubstitutions(command, ...
                                                 isDMCRemote, localDMCRootDir, ...
                                                 didUploadMovies, localPathFromMovieIndex, remotePathFromMovieIndex)
      % Replate the local DMCoD root with the remote one
      if isDMCRemote ,
        result_1 = strrep(command, localDMCRootDir, AWSec2.remoteDLCacheDir) ;
      else
        result_1 = command ;
      end      
      % Replace local movie paths with the corresponding remote ones
      if didUploadMovies ,
        result_2 = strrep_multiple(result_1, localPathFromMovieIndex, remotePathFromMovieIndex) ;
      else
        result_2 = result_1 ;
      end
      % Replace the local APT source root with the remote one
      remote_apt_root = AWSec2.remoteAPTSourceRootDir ;
      local_apt_root = APT.Root ;
      result_3 = strrep(result_2, local_apt_root, remote_apt_root) ;
      % Replace the local home dir with the remote one
      % Do this last b/c e.g. the local APT source root is likely in the local home
      % dir.
      local_home_path = get_home_dir_name() ;
      remote_home_path = AWSec2.remoteHomeDir ;
      result_4 = strrep(result_3, local_home_path, remote_home_path) ;      
      result = result_4 ;
    end  % function
  end  % methods (Static)
end  % classdef
