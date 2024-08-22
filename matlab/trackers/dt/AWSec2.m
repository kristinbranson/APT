classdef AWSec2 < matlab.mixin.Copyable
  % Handle to a single AWS EC2 instance. The instance may be in any state,
  % running, stopped, etc.
  %
  % An AWSec2's .instanceID property is immutable. If you change it, it is 
  % a different EC2 instance! So .instanceID is either [], indicating an 
  % "unspecified instance", or it is specified to some fixed value. Rather 
  % than attempt to mutate a specified instance, just create a new AWSec2
  % object.
  %
  % Some methods below can specify (assign an .instanceID) to an 
  % unspecified instance. No method however will mutate the .instanceID of 
  % a specified instance.

  properties (Constant)
    autoShutdownAlarmNamePat = 'aptAutoShutdown'; 
  end
  
  properties (SetAccess=private)
    instanceID  % primary ID. depending on config, IPs can change when instances are stopped/restarted etc.
  end

  properties (Dependent)
    isSpecified
    isConfigured
    isInDebugMode
  end

  properties
    instanceIP
    keyName = ''
    pem = ''
    instanceType = 'p3.2xlarge';
    remotePID
    %SetStatusFun = @(s,varargin) fprintf(['AWS Status: ',s,'\n'])
    %ClearStatusFun = @(varargin) fprintf('Done.\n')
  end
  
  properties (Transient)
    isInDebugMode_ = false
  end
  
  properties (Constant)
    cmdEnv = 'sleep 5 ; LD_LIBRARY_PATH= AWS_PAGER='  
      % Change the sleep number at your own risk!  I tried 3, and everything seemed
      % fine for a while until it became a very hard-to-find bug. -- ALT, 2024-06-28
    scpCmd = AWSec2.computeScpCmd()
    sshCmd = AWSec2.computeSshCmd()
    rsyncCmd = AWSec2.computeRsyncCmd()
  end
  
  methods
  end
  
  methods
    
    function obj = AWSec2(varargin)
      
%       if ispc()
%         windows_null_device_path = '\\.\NUL' ;
%         obj.scpCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSCPCMD, windows_null_device_path) ; 
%         obj.sshCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSSHCMD, windows_null_device_path) ; 
%       else
%         obj.scpCmd = 'scp -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR';
%         obj.sshCmd = 'ssh -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR';
%       end

      for i=1:2:numel(varargin)
        prop = varargin{i};
        val = varargin{i+1};
        obj.(prop) = val;
      end
      
    end
    
    function delete(obj)  %#ok<INUSD> 
      % NOTE: for now, lifecycle of obj is not tied at all to the actual
      % instance-in-the-cloud
    end
    
  end
      % NOTE: for now, lifecycle of obj is not tied at all to the actual
      % instance-in-the-cloud

  methods    
    function set.instanceID(obj,v)
      obj.instanceID = v;
    end

    function v = get.isConfigured(obj)
      v = ~isempty(obj.pem) && ~isempty(obj.keyName);
    end

    function v = get.isSpecified(obj)
      v = ~isempty(obj.instanceID);
    end

    function result = get.isInDebugMode(obj)
      result = obj.isInDebugMode_ ;
    end

    function set.isInDebugMode(obj, value)
      obj.isInDebugMode_ = value ;
    end

%     function SetStatus(obj,varargin)
% %       if ~isempty(obj.SetStatusFun),
% %         obj.SetStatusFun(varargin{:});
% %       end
%     end
% 
%     function ClearStatus(obj,varargin)
% %       if ~isempty(obj.ClearStatusFun),
% %         obj.ClearStatusFun(varargin{:});
% %       end
%     end
% 
%     function clearStatusFuns(obj)
% %       % AL20191218
% %       % AWSEc2 objects are deep-copied onto bg worker objects and this is
% %       % causing problems on Linux because these function handles contain
% %       % references to the Labeler and the entire GUI is being
% %       % serialized/copied. 
% %       %
% %       % For these worker objects these statusFun handles are unnec and we 
% %       % clear them out.
% %       obj.SetStatusFun = [];
% %       obj.ClearStatusFun = [];
%     end
    
    function setInstanceID(obj,instanceID,instanceType)
      %obj.SetStatus(sprintf('Setting AWS EC2 instance = %s',instanceID));
      if ~isempty(obj.instanceID),
        if strcmp(instanceID,obj.instanceID),
          % nothing to do
          %obj.ClearStatus();
          return;
        end
        %instanceID = obj.instanceID;
        [tfexist,tfrunning,json] = obj.inspectInstance();
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
      if obj.isSpecified,
        obj.configureAlarm();
      end
      %obj.ClearStatus();
    end

    function setPemFile(obj,pemFile)
      obj.pem = pemFile;
    end
    
    function setKeyName(obj,keyName)
      obj.keyName = keyName;
    end
    
    function [tfsucc,json] = launchInstance(obj,varargin)
      % Launch a brand-new instance to specify an unspecified instance
      [dryrun,dostore] = myparse(varargin,'dryrun',false,'dostore',true);
      obj.ResetInstanceID();
      %obj.SetStatus('Launching new AWS EC2 instance');
      cmd = AWSec2.launchInstanceCmd(obj.keyName,'instType',obj.instanceType,'dryrun',dryrun);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        obj.ClearStatus();
        return;
      end
      json = jsondecode(json);
      instanceID = json.Instances.InstanceId;
      if dostore,
        obj.setInstanceID(instanceID);
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
      
      dispcmd = myparse(varargin,...
        'dispcmd',true...
        );
      
      assert(obj.isSpecified,'Cannot inspect an unspecified AWSEc2 instance.');
      
      % Aside: this works with empty .instanceID if there is only one 
      % instance in the cloud, but we are not interested in that for now
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); 
      [tfexist,json] = AWSec2.syscmd(cmd,'dispcmd',dispcmd,'isjsonout',true);
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
      assert(obj.isSpecified);
      state = '';
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      state = json.Reservations.Instances.State.Name;      
    end
    
    function [tfsucc,instanceID,pemFile] = respecifyInstance(obj)
      [tfsucc,instanceID,pemFile] = ...
        obj.specifyInstanceUIStc(obj.instanceID,obj.pem);
    end
    
    function [tfsucc,keyName,pemFile] = specifyPemKeyType(obj,dostore)
      if nargin < 2,
        dostore = false;
      end
      [tfsucc,keyName,pemFile] = ...
        obj.specifySSHKeyUIStc(obj.keyName,obj.pem);
      if tfsucc && dostore,
        obj.setPemFile(pemFile);
        obj.setKeyName(keyName);
      end
    end
    
    function [tfsucc,json] = stopInstance(obj,varargin)
      [isinteractive] = myparse(varargin,'isinteractive',false);
      if ~obj.isSpecified,
        tfsucc = true;
        json = {};
        return;
      end
      if isinteractive,
        res = questdlg(sprintf('Stop AWS instance %s? If you stop your instance, running other computations on AWS will have some overhead as the instance is re-initialized. If you do not stop the instance now, you may need to manually stop the instance in the future.',obj.instanceID),'Stop AWS Instance');
        if ~strcmpi(res,'Yes'),
          tfsucc = false;
          return;
        end
      end
      %obj.SetStatus(sprintf('Stopping AWS EC2 instance %s',obj.instanceID));
      cmd = AWSec2.stopInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      %obj.ClearStatus();
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      obj.stopAlarm();
    end  % function
    

    function [tfsucc,instanceIDs,instanceTypes,json] = listInstances(obj)
    
      instanceIDs = {};
      instanceTypes = {};
      %obj.SetStatus('Listing AWS EC2 instances available');
      cmd = AWSec2.listInstancesCmd(obj.keyName,'instType',[]); % empty instType to list all instanceTypes
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if tfsucc,
        info = jsondecode(json);
        if ~isempty(info.Reservations),
          instanceIDs = arrayfun(@(x)x.Instances.InstanceId,info.Reservations,'uni',0);
          instanceTypes = arrayfun(@(x)x.Instances.InstanceType,info.Reservations,'uni',0);
        end
      end
      %obj.ClearStatus();
      
    end

    function [tfsucc,json,warningstr,state] = startInstance(obj,varargin)

      %obj.SetStatus(sprintf('Starting instance %s',obj.instanceID));
      [doblock] = myparse(varargin,'doblock',true);
      
      maxwaittime = 100;
      iterwaittime = 5;
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
        [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      end
      if ~tfsucc
        %obj.ClearStatus();
        return;
      end
      json = jsondecode(json);
      if ~doblock,
        %obj.ClearStatus();
        return;
      end
      
      [tfsucc] = obj.waitForInstanceStart();
      if ~tfsucc,
        warningstr = 'Timed out waiting for AWS EC2 instance to spool up.';
        %obj.ClearStatus();
        return;
      end
      
      obj.inspectInstance();
      obj.configureAlarm();
      %obj.ClearStatus();
    end
    
    function [tfsucc] = waitForInstanceStart(obj)
      
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
              [tfsucc,res] = obj.cmdInstance('cat /dev/null','dispcmd',true);
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
      
    end
    
    function checkInstanceRunning(obj,varargin)
      % - If runs silently, obj appears to be a running EC2 instance with 
      %   no issues
      % - If harderror thrown, something appears wrong
      %obj.SetStatus('Checking whether AWS EC2 instance is running');
      throwErrs = myparse(varargin,...
        'throwErrs',true... % if false, just warn if there is a problem
        );
      
      if throwErrs
        throwFcn = @error;
      else
        throwFcn = @warningNoTrace;
      end
      
      [tfexist,tfrun] = obj.inspectInstance;
      %obj.ClearStatus();

      if ~tfexist
        throwFcn('Problem with EC2 instance id: %s',obj.instanceID);
      end
      if ~tfrun
        throwFcn('EC2 instance id %s is not in the ''running'' state.',...
          obj.instanceID)
      end

    end
    
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
      
      [tfe,~,js] = obj.inspectInstance('dispcmd',false);
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
      if ~obj.isSpecified,
        reason = 'AWS EC2 instance not specified.';
        return;
      end
      namestr = sprintf(obj.autoShutdownAlarmNamePat);

      codestr = sprintf('aws cloudwatch describe-alarms --alarm-names "%s"',namestr);
      [tfsucc,json] = obj.syscmd(codestr,...
        'dispcmd',true,...
        'failbehavior','warn',...
        'isjsonout',true);
      if ~tfsucc,
        reason = 'AWS CLI error calling describe-alarms.';
        return;
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
      
      % TODO
      
    end
    
    function tfsucc = configureAlarm(obj)
      % Note: this creates/puts a metricalarm with name based on the
      % instanceID. Currently the AWS API allows you to create the same
      % alarm multiple times with no harm (only one alarm is ultimately
      % created).

      [tfsucc,isalarm,reason] = obj.checkShutdownAlarm();
      if ~tfsucc,
        warning('Could  not check for alarm: %s',reason);
        return;
      end
      % DEBUGAWS: AWS alarms are annoying while debugging
      if obj.isInDebugMode_ ,
        return
      end
      if isalarm,
        return
      end
      
      codestr = obj.createShutdownAlarmCmd ;
      
      fprintf('Setting up AWS CloudWatch alarm to auto-shutdown your instance if it is idle for too long.\n');
      
      tfsucc = obj.syscmd(codestr,...
                          'dispcmd',true,...
                          'failbehavior','warn');      
    end
    
    function tfsucc = getRemotePythonPID(obj)
      [tfsucc,res] = obj.cmdInstance('pgrep --uid ubuntu --oldest python','dispcmd',true);
      if tfsucc
        pid = str2double(strtrim(res));
        obj.remotePID = pid; % right now each aws instance only has one GPU, so can only do one train/track at a time
        fprintf('Remote PID is: %d.\n\n',pid);
      else
        warningNoTrace('Failed to ascertain remote PID.');
      end
    end
    
    function tfnopyproc = getNoPyProcRunning(obj)
      % Return true if there appears to be no python process running on
      % instance
      [tfsucc,res] = obj.cmdInstance('pgrep --uid ubuntu --oldest python',...
        'dispcmd',true,'failbehavior','silent');
      
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
        tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
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
        'sshcmd',obj.sshCmd,'destRelative',destRelative);
      AWSec2.syscmd(cmd,sysCmdArgs{:});
      cmd = AWSec2.scpUploadCmd(file,obj.pem,obj.instanceIP,dest,...
        'scpcmd',obj.scpCmd,'destRelative',destRelative);
      tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
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
      tfsucc = obj.remoteFileExists(dstAbs,'dispcmd',true,'size',src_sz);
      if tfsucc
        fprintf('%s file exists: %s.\n\n',...
          String.niceUpperCase(fileDescStr),dstAbs);
      else
        %obj.SetStatus(sprintf('Uploading %s file to AWS EC2 instance',fileDescStr));
        fprintf('About to upload. This could take a while depending ...\n');
        tfsucc = obj.scpUpload(src,dstAbs,...
          'destRelative',false,'sysCmdArgs',{'dispcmd',true});
        %obj.ClearStatus();
        if tfsucc
          fprintf('Uploaded %s %s to %s.\n\n',fileDescStr,src,dst);
        else
          error('Failed to upload %s %s.',fileDescStr,src);
        end
      end
    end
    
    function scpUploadOrVerifyEnsureDir(obj,fileLcl,fileRemote,fileDescStr,...
        varargin)
      % Upload a file to a dir which may not exist yet. Create it if 
      % necessary. Either succeeds, or fails and harderrors.
      
      destRelative = myparse(varargin,...
        'destRelative',false);
      
      dirRemote = fileparts(fileRemote);
      obj.ensureRemoteDir(dirRemote,'relative',destRelative); 
      obj.scpUploadOrVerify(fileLcl,fileRemote,fileDescStr,...
        'destRelative',destRelative); % throws
    end
       
    function tfsucc = rsyncUpload(obj, src, dest)
      cmd = AWSec2.rsyncUploadCmd(src, obj.pem, obj.instanceIP, dest) ;
      tfsucc = AWSec2.syscmd(cmd) ;
    end

    function rmRemoteFile(obj,dst,fileDescStr,varargin) % throws
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
      [tfsucc,res] = obj.cmdInstance(cmd,'dispcmd',true,'failbehavior','err'); %#ok<ASGLU>
      %obj.ClearStatus();
    end
    
    
    function tf = remoteFileExists(obj,f,varargin)
      [reqnonempty,dispcmd,usejavaRT,size] = myparse(varargin,...
        'reqnonempty',false,...
        'dispcmd',false,...
        'usejavaRT',false,...
        'size',-1 ...
        );

      if reqnonempty
        script = '~/APT/matlab/misc/fileexistsnonempty.sh';
      else
        script = '~/APT/matlab/misc/fileexists.sh';
      end
      if size > 0
        cmdremote = sprintf('%s %s %d',script,f,size);
      else
        cmdremote = sprintf('%s %s',script,f);
      end
      %logger.log('AWSSec2::remoteFileExists() milestone 1\n') ;
      [~,res] = obj.cmdInstance(cmdremote,'dispcmd',dispcmd,'failbehavior','err','usejavaRT',usejavaRT); 
      %logger.log('AWSSec2::remoteFileExists() milestone 2.  status=%d\nres=\n%s\n', status, res) ;
      tf = (res(1)=='y');      
    end
    
    function s = remoteFileContents(obj,f,varargin)
      [dispcmd,failbehavior] = myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn'...
        );
      
      cmdremote = sprintf('cat %s',f);
      [tfsucc,res] = obj.cmdInstance(cmdremote, ...
                                     'dispcmd',dispcmd, ...
                                     'failbehavior',failbehavior); 
      if tfsucc  
        s = res;
      else
        % warning thrown etc per failbehavior
        s = '';
      end
    end
    
    function result = remoteFileModTime(obj, filename, varargin)
      % Returns the file modification time (mtime) in seconds since Epoch
      [dispcmd, failbehavior] = ...
        myparse(varargin, ...
                'dispcmd', false, ...
                'failbehavior', 'warn') ;
      command = sprintf('stat --format=%%Y %s', escape_string_for_bash(filename)) ;  % time of last data modification, seconds since Epoch
      [did_succeed, stdouterr] = obj.cmdInstance(command, 'dispcmd', dispcmd, 'failbehavior', failbehavior) ; 
      if did_succeed ,
        result = str2double(stdouterr) ;
      else
        % Warning/error happens inside obj.cmdInstance(), so just set a fallback value
        result = nan ;
      end
    end
    
    function tfsucc = remoteLS(obj,remoteDir,varargin)
      [dispcmd,failbehavior,args] = myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn',...
        'args','-lha'...
        );
      
      cmdremote = sprintf('ls %s %s',args,remoteDir);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',dispcmd,...
        'failbehavior',failbehavior);
      
      disp(res);
      % warning thrown etc per failbehavior
    end
    
    function remoteDirFull = ensureRemoteDir(obj,remoteDir,varargin)
      % Creates/verifies remote dir. Either succeeds, or fails and harderrors.
      
      [relative,descstr] = myparse(varargin,...
        'relative',true,...  % true if remoteDir is relative to ~
        'descstr',''... % cosmetic, for disp/err strings
        );
      
      if ~isempty(descstr)
        descstr = [descstr ' '];
      end

      if relative
        remoteDirFull = ['~/' remoteDir];
      else
        remoteDirFull = remoteDir;
      end
      
      %obj.SetStatus(sprintf('Creating directory %s on AWS EC2 instance',remoteDirFull));
      cmdremote = sprintf('mkdir -p %s',remoteDirFull);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
      %obj.ClearStatus();
      if tfsucc
        fprintf('Created/verified remote %sdirectory %s: %s\n\n',...
          descstr,remoteDirFull,res);
      else
        error('Failed to create remote %sdirectory %s: %s',descstr,...
          remoteDirFull,res);
      end
    end
    
    function remotePaths = remoteGlob(obj,globs)
      % Look for remote files/paths. Either succeeds, or fails and harderrors.

      % globs: cellstr of globs
      
      lscmd = cellfun(@(x)sprintf('ls %s 2> /dev/null;',x),globs,'uni',0);
      lscmd = cat(2,lscmd{:});
      [tfsucc,res] = obj.cmdInstance(lscmd);
      if tfsucc
        remotePaths = regexp(res,'\n','split');
        remotePaths = remotePaths(:);
        remotePaths = remotePaths(~cellfun(@isempty,remotePaths));
      else
        error('Failed to find remote files/paths %s: %s',...
          String.cellstr2CommaSepList(globs),res);
      end      
    end
    
    function cmdfull = cmdInstanceDontRun(obj, cmdremote, varargin)
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd, obj.pem, obj.instanceIP, cmdremote, 'usedoublequotes', true) ;
    end

    function [tfsucc,res,cmdfull] = cmdInstance(obj,cmdremote,varargin)
      % Runs a single command-line command on the ec2 instance
%       [logger] = myparse(varargin,...
%                          'logger',FileLogger());
%       logger.log('cmdInstance: cmdremote: %s\n', cmdremote) ;
      fprintf('cmdInstance: %s\n',cmdremote);
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd, obj.pem, obj.instanceIP, cmdremote, 'usedoublequotes', true) ;
%       logger.log('cmdInstance: cmdfull: %s\n', cmdfull) ;
      [tfsucc,res] = AWSec2.syscmd(cmdfull, varargin{:}) ;
%       logger.log('cmdInstance: tfsucc: %d, res: \n%s\n', tfsucc, res) ;
    end
        
    function cmd = sshCmdGeneralLogged(obj, cmdremote, logfileremote)
      cmd = sprintf('%s -i %s ubuntu@%s "%s </dev/null >%s 2>&1 &"',...
                    obj.sshCmd, obj.pem, obj.instanceIP, cmdremote, logfileremote) ;
    end
        
    function tf = canKillRemoteProcess(obj)
      tf = ~isempty(obj.remotePID) && ~isnan(obj.remotePID);
    end
    
    function killRemoteProcess(obj)
      % AL 20200213: now do this in a loop, kill until no py processes
      % remain. For some nets (eg LEAP) multiple py (sub)procs are spawned
      % and not sure a single kill does the job.

%       if isempty(obj.remotePID)
%         error('Unknown PID for remote process.');
%       end
%       
%       cmdremote = sprintf('kill %d',obj.remotePID);
      cmdremote = 'pkill --uid ubuntu --full python';
      [tfsucc,~] = obj.cmdInstance(cmdremote,'dispcmd',true);
      if tfsucc
        fprintf('Kill command sent.\n\n');
      else
        error('Kill command failed.');
      end
    end

    function [tfsucc,res] = remoteCallFSPoll(obj,fspollargs)
      % fspollargs: [n] cellstr eg {'exists' '/my/file' 'existsNE' '/my/file2'}
      %
      % res: [n] cellstr of fspoll responses

      assert(iscellstr(fspollargs) && ~isempty(fspollargs));  %#ok<ISCLSTR> 
      nargsFSP = numel(fspollargs);
      assert(mod(nargsFSP,2)==0);
      nresps = nargsFSP/2;
      
      fspollstr = sprintf('%s ',fspollargs{:});
      fspollstr = fspollstr(1:end-1);
      cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollstr);

      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
      if tfsucc
        res = regexp(res,'\n','split');
        tfsucc = iscell(res) && numel(res)==nresps+1; % last cell is {0x0 char}
        res = res(1:end-1);
      else
        res = [];
      end
    end
    
    function ResetInstanceID(obj)
      obj.setInstanceID('');
    end
    
%     function tf = isSameInstance(obj,obj2)
%       assert(isscalar(obj) && isscalar(obj2));
%       tf = strcmp(obj.instanceID,obj2.instanceID) && ~isempty(obj.instanceID);
%     end
  end  % methods
  
  methods (Static)
    
    function cmd = launchInstanceCmd(keyName,varargin)
      [ami,instType,secGrp,dryrun] = myparse(varargin,...
        'ami',APT.AMI,...
        'instType','p3.2xlarge',...
        'secGrp',APT.AWS_SECURITY_GROUP,...
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
      
      [ami,instType,secGrp] = myparse(varargin,...
        'ami',APT.AMI,...
        'instType','p3.2xlarge',...
        'secGrp',APT.AWS_SECURITY_GROUP,...
        'dryrun',false);
      
      cmd = sprintf('aws ec2 describe-instances --filters "Name=image-id,Values=%s" "Name=instance.group-name,Values=%s" "Name=key-name,Values=%s"',ami,secGrp,keyName);
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
    
    function [tfsucc,res,warningstr] = syscmd(cmd,varargin)
      % Run a command using Matlab built-in system() command.
      % Optional args allow caller to specify what to do if the command fails.  (Can
      % error, warn, or silently ignore.)  Also has option to lightly process a JSON
      % response.
      %
      % Seems to also support issuing the command using the Java runtime, but AFAICT
      % this is never used in APT.  -- ALT, 2024-04-25
      [dispcmd,failbehavior,isjsonout,dosetenv,setenvcmd,usejavaRT] = ...
        myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn',... % one of 'err','warn','silent'
        'isjsonout',false,...
        'dosetenv',isunix(),...
        'setenvcmd',AWSec2.cmdEnv,...
        'usejavaRT',false...
        );
      
%       cmd = [cmd sprintf('\n\r')];
      if dosetenv,
        cmd = [setenvcmd,' ',cmd];
      end

      % XXX HACK
      drawnow 

      if dispcmd
        disp(cmd); 
      end
      if usejavaRT
        fprintf(1,'Using javaRT call\n');
        runtime = java.lang.Runtime.getRuntime();
        proc = runtime.exec(cmd);
        st = proc.waitFor();
        is = proc.getInputStream;
        res = [];
        val = is.read();
        while val~=-1 && numel(res)<100
          res(1,end+1) = val;  %#ok<AGROW> 
          val = is.read();
        end
        res = strtrim(char(res));
        tfsucc = st==0;
      else
        fprintf('syscmd: %s\n',cmd);
        [st,res] = system(cmd);
        if st ~= 0,
          fprintf('st = %d, res = %s\n',st,res);
        else
          fprintf('success.\n');
        end
        tfsucc = st==0 || isempty(res);
      end
      
      if isjsonout && tfsucc,
        jsonstart = find(res == '{',1);
        if isempty(jsonstart),
          tfsucc = false;
          warningstr = 'Could not find json start character {';
        else
          warningstr = res(1:jsonstart-1);
          res = res(jsonstart:end);
        end
      else
        warningstr = '';
      end
      
      if ~tfsucc 
        switch failbehavior
          case 'err'
            error('Nonzero status code: %s',res);
          case 'warn'
            warningNoTrace('Command failed: %s: %s',cmd,res);
          case 'silent'
            % none
          otherwise
            assert(false);
        end
      end
    end  % syscmd()
    
    function cmd = scpPrepareUploadCmd(pem,ip,dest,varargin)
      [destRelative,sshcmd] = myparse(varargin,...
        'destRelative',true,...
        'sshcmd','ssh');
      if destRelative
        dest = ['~/' dest];
      end
      [parentdir] = fileparts(dest);
      cmdremote = sprintf('[ ! -d %s ] && mkdir -p %s',parentdir,parentdir);
      cmd = AWSec2.sshCmdGeneral(sshcmd,pem,ip,cmdremote,'usedoublequotes',true);      
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
      sshcmd = sprintf('%s -o ConnectTimeout=8 -i %s', AWSec2.sshCmd, pemFilePath) ;
      escaped_sshcmd = escape_string_for_bash(sshcmd) ;

      % Generate the final command
      cmd = sprintf('%s --rsh=%s %s/ ubuntu@%s:%s', AWSec2.rsyncCmd, escaped_sshcmd, src, ip, dest) ;
    end

    function cmd = sshCmdGeneral(sshcmd, pem, ip, cmdremote, varargin)
      [timeout,~] = myparse(varargin,...
                            'timeout',8,...
                            'usedoublequotes',false);
      
      args = { sshcmd '-i' pem sprintf('-o ConnectTimeout=%d', timeout) sprintf('ubuntu@%s',ip) } ;
      args{end+1} = escape_string_for_bash(cmdremote) ;
%       if usedoublequotes
%         args{end+1} = sprintf('"%s"',cmdremote);
%       else
%         args{end+1} = sprintf('''%s''',cmdremote);
%       end
      cmd = String.cellstr2DelimList(args,' ');
    end  % function

    function [tfsucc,instanceID,pemFile] = ...
                              specifyInstanceUIStc(instanceID,pemFile)
      % Prompt user to specify/confirm an AWS instance.
      % 
      % instanceID, pemFile (in): optional defaults/best guesses
      
      if nargin<1
        instanceID = '';
      end
      if nargin<2
        pemFile = '';
      end
      
      PROMPT = {
        'Instance ID'
        'Private key (.pem) file'
        };
      NAME = 'AWS EC2 Config';
      INPUTBOXWIDTH = 100;
      BROWSEINFO = struct('type',{'';'uigetfile'},'filterspec',{'';'*.pem'});

      resp = inputdlgWithBrowse(PROMPT,NAME,repmat([1 INPUTBOXWIDTH],2,1),...
        {instanceID;pemFile},'on',BROWSEINFO);
      tfsucc = ~isempty(resp);      
      if tfsucc
        instanceID = strtrim(resp{1});
        pemFile = strtrim(resp{2});
        if exist(pemFile,'file')==0
          error('Cannot find private key (.pem) file %s.',pemFile);
        end
      else
        instanceID = [];
        pemFile = [];
      end      
    end
    
    function [tfsucc,keyName,pemFile] = ...
        specifySSHKeyUIStc(keyName,pemFile)
      % Prompt user to specify pemFile
      % 
      % keyName, pemFile (in): optional defaults/best guesses
      
      if nargin<1 || isempty(keyName),
        keyName = '';
      end
      if nargin<2 || isempty(pemFile),
        pemFile = '';
      end
      
      PROMPT = {
        'Key name'
        'Private key (.pem or id_rsa) file'
        };
      NAME = 'AWS EC2 Config';
      INPUTBOXWIDTH = 100;
      BROWSEINFO = struct('type',{'';'uigetfile'},'filterspec',{'';'*.pem'});

      resp = inputdlgWithBrowse(PROMPT,NAME,repmat([1 INPUTBOXWIDTH],2,1),...
        {keyName;pemFile},'on',BROWSEINFO);
      tfsucc = ~isempty(resp);      
      if tfsucc
        keyName = strtrim(resp{1});
        pemFile = strtrim(resp{2});
        if exist(pemFile,'file')==0
          error('Cannot find private key (.pem or id_rsa) file %s.',pemFile);
        end
      else
        keyName = '';
        pemFile = '';
      end      
    end  % function

    function scpCmd = computeScpCmd()
      if ispc()
        windows_null_device_path = '\\.\NUL' ;
        scpCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSCPCMD, windows_null_device_path) ; 
      else
        scpCmd = 'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
      end
    end

    function sshCmd = computeSshCmd()
      if ispc()
        windows_null_device_path = '\\.\NUL' ;
        sshCmd = sprintf('"%s" -oStrictHostKeyChecking=no -oUserKnownHostsFile=%s -oLogLevel=ERROR', APT.WINSSHCMD, windows_null_device_path) ; 
      else
        sshCmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR';
      end
    end
    
    function result = computeRsyncCmd()
      if ispc()
        error('Not implemented') ;
      else
        result = 'rsync -az' ;
      end
    end
    
  end  % Static methods block
  
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
  
end