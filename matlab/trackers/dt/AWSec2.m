classdef AWSec2 < handle
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
    instanceID % primary ID. depending on config, IPs can change when instances are stopped/restarted etc.    
  end
  properties (Dependent)
    isSpecified
    isConfigured
  end
  properties
    instanceIP
    keyName = '';
    pem = '';

    instanceType = 'p2.xlarge';
    
    scpCmd
    sshCmd
    
    remotePID
    
    SetStatusFun = @(s,varargin) fprintf(['AWS Status: ',s,'\n']);
    ClearStatusFun = @(varargin) fprintf('Done.\n');

  end
  
  properties (Constant)
    cmdEnv = 'LD_LIBRARY_PATH=: ';
  end
  
  methods
    function set.instanceID(obj,v)
%       if ~isempty(obj.instanceID) && ~strcmp(v,obj.instanceID),
%         fprintf('AWSEc2 instanceID was already set to %s, overwriting to %s.',obj.instanceID,v);
%       end
      obj.instanceID = v;
    end
    function v = get.isConfigured(obj)
      v = ~isempty(obj.pem) && ~isempty(obj.keyName);
    end
    function v = get.isSpecified(obj)
      v = ~isempty(obj.instanceID);
    end
  end
  
  methods
    
    function obj = AWSec2(varargin)
      
      if nargin >= 1,
        pem = varargin{1};
        if ~isempty(pem),
          obj.pem = pem;
        end
      end
            
      if ispc
        obj.scpCmd = ['"',APT.WINSCPCMD,'"'];
        obj.sshCmd = ['"',APT.WINSSHCMD,'"'];
      else
        obj.scpCmd = 'scp';
        obj.sshCmd = 'ssh';
      end

      for i=2:2:numel(varargin)
        prop = varargin{i};
        val = varargin{i+1};
        obj.(prop) = val;
      end
      
    end
    
    function delete(obj)
      % NOTE: for now, lifecycle of obj is not tied at all to the actual
      % instance-in-the-cloud
    end
    
  end
      % NOTE: for now, lifecycle of obj is not tied at all to the actual
      % instance-in-the-cloud

  methods
    
    function SetStatus(obj,varargin)
      if ~isempty(obj.SetStatusFun),
        obj.SetStatusFun(varargin{:});
      end
    end
    function ClearStatus(obj,varargin)
      if ~isempty(obj.ClearStatusFun),
        obj.ClearStatusFun(varargin{:});
      end
    end
    
    function setInstanceID(obj,instanceID,instanceType)

      obj.SetStatus(sprintf('Setting AWS EC2 instance = %s',instanceID));

      if ~isempty(obj.instanceID),
        if strcmp(instanceID,obj.instanceID),
          % nothing to do
          obj.ClearStatus();
          return;
        end
        %instanceID = obj.instanceID;
        [tfexist,tfrunning,json] = obj.inspectInstance();
        if tfexist && tfrunning,
          tfsucc = obj.stopInstance();
          if ~tfsucc,
            warning('Error stopping old AWS EC2 instance %s.',instanceID);
          end
          obj.SetStatus(sprintf('Setting AWS EC2 instance = %s',instanceID));
        end
      end
      obj.instanceID = instanceID;
      if nargin > 3,
        obj.instanceType = instanceType;
      end
      if obj.isSpecified,
        obj.configureAlarm();
      end
      obj.ClearStatus();
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
      
%       assert(~obj.isSpecified,...
%         'AWSEc2 instance is already specified with instanceID %s.',obj.instanceID);
      
      
      obj.SetStatus('Launching new AWS EC2 instance');
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
      obj.SetStatus('Waiting for AWS EC2 instance to spool up.');
      [tfsucc] = obj.waitForInstanceStart();
      if ~tfsucc,
        obj.ClearStatus();
        return;
      end
      obj.configureAlarm();
      obj.ClearStatus();
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
    end
    
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
      
      %[tfsucc,instanceID,instanceType,reason] = obj.selectInstance('dostore',false);
      
      [tfsucc,instanceID,pemFile] = ...
        obj.specifyInstanceUIStc(obj.instanceID,obj.pem,'instanceIDs',instanceIDs,'instanceTypes',instanceTypes);
    end
    
    function [tfsucc,keyName,pemFile] = respecifySSHKey(obj,dostore)
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
      obj.SetStatus(sprintf('Stopping AWS EC2 instance %s',obj.instanceID));
      cmd = AWSec2.stopInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      obj.ClearStatus();
      if ~tfsucc
        return;
      end
      json = jsondecode(json);

      obj.stopAlarm();
      
    end
    

    function [tfsucc,instanceIDs,instanceTypes,json] = listInstances(obj)
    
      instanceIDs = {};
      instanceTypes = {};
      obj.SetStatus('Listing AWS EC2 instances available');
      cmd = AWSec2.listInstancesCmd(obj.keyName);%,'instType',obj.instanceType);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if tfsucc,
        info = jsondecode(json);
        if ~isempty(info.Reservations),
          instanceIDs = {info.Reservations.Instances.InstanceId};
          instanceTypes = {info.Reservations.Instances.InstanceType};
        end
      end
      obj.ClearStatus();
      
    end
    
    function [tfsucc,instanceID,instanceType,reason,didLaunch] = selectInstance(obj,varargin)

      [canLaunch,canConfigure,forceSelect] = ...
        myparse(varargin,'canlaunch',true,...
        'canconfigure',1,'forceSelect',true);
      
      reason = '';
      instanceID = '';
      instanceType = '';
      didLaunch = false;
      tfsucc = false;
      
      if ~obj.isConfigured || canConfigure >= 2,
        if canConfigure,
          [tfsucc] = obj.respecifySSHKey(true);
          if ~tfsucc && ~obj.isConfigured,
            reason = 'AWS EC2 instance is not configured.';
            return;
          else
            tfsucc = true;
          end
        else
          reason = 'AWS EC2 instance is not configured.';
          return;
        end
      end
      if forceSelect || ~obj.isSpecified,
        if obj.isSpecified,
          instanceID = obj.instanceID;
        else
          instanceID = '';
        end
        if canLaunch,
          qstr = 'Launch a new instance or attach to an existing instance?';
          if ~obj.isSpecified,
            qstr = ['APT is not attached to an AWS EC2 instance. ',qstr];
          else
            qstr = sprintf('APT currently attached to AWS EC2 instance %s. %s',instanceID,qstr);
          end
          tstr = 'Specify AWS EC2 instance';
          btn = questdlg(qstr,tstr,'Launch New','Attach to Existing','Cancel','Cancel');
          if isempty(btn)
            btn = 'Cancel';
          end
        else
          btn = 'Attach to Existing';
        end
        while true,
          switch btn
            case 'Launch New'
              tf = obj.launchInstance();
              if ~tf
                reason = 'Could not launch AWS EC2 instance.';
                return;
              end
              instanceID = obj.instanceID;
              instanceType = obj.instanceType;
              didLaunch = true;
              break;
            case 'Attach to Existing',

              [tfsucc,instanceIDs,instanceTypes] = obj.listInstances();
              if ~tfsucc,
                reason = 'Error listing instances.';
                return;
              end
              if isempty(instanceIDs),
                if canLaunch,
                  btn = questdlg('No instances found. Launch a new instance?',tstr,'Launch New','Cancel','Cancel');
                  continue;
                else
                  tfsucc = false;
                  reason = 'No instances found.';
                  return;
                end
              end
              
              PROMPT = {
                'Instance'
                };
              NAME = 'AWS EC2 Select Instance';
              INPUTBOXWIDTH = 100;
              BROWSEINFO = struct('type',{'popupmenu'});
              s = cellfun(@(x,y) sprintf('%s (%s)',x,y),instanceIDs,instanceTypes,'Uni',false);
              v = 1;
              if ~isempty(obj.instanceID),
                v = find(strcmp(instanceIDs,obj.instanceID),1);
                if isempty(v),
                  v = 1;
                end
              end
              DEFVAL = {{s,v}};
              resp = inputdlgWithBrowse(PROMPT,NAME,repmat([1 INPUTBOXWIDTH],1,1),...
                DEFVAL,'on',BROWSEINFO);
              tfsucc = ~isempty(resp);
              if tfsucc
                instanceID = instanceIDs{resp{1}};
                instanceType = instanceTypes{resp{1}};
              else
                reason = 'Canceled.';
                return;
              end
              break;
            otherwise
              reason = 'Canceled.';
              return;
          end
        end
        obj.setInstanceID(instanceID,instanceType);
%         obj.instanceID = instanceID;
%         obj.instanceType = instanceType;
      end
      tfsucc = true;

    end
    
    function [tfsucc,json,warningstr,state] = startInstance(obj,varargin)

      obj.SetStatus(sprintf('Starting instance %s',obj.instanceID));
      [doblock] = myparse(varargin,'doblock',true);
      
      maxwaittime = 100;
      iterwaittime = 5;
      warningstr = '';
      [tfsucc,state,json] = obj.getInstanceState();
      if ~tfsucc,
        warningstr = 'Failed to get instance state.';
        obj.ClearStatus();
        return;
      end

      if ismember(lower(state),{'shutting-down','terminated'}),
        warningstr = sprintf('Instance is %s, cannot start',state);
        tfsucc = false;
        obj.ClearStatus();
        return
      end
      if ismember(lower(state),{'stopping'}),
        warningstr = sprintf('Instance is %s, please wait for this to finish before starting.',state);
        tfsucc = false;
        obj.ClearStatus();
        return;
      end
      if ~ismember(lower(state),{'running','pending'}),
        cmd = AWSec2.startInstanceCmd(obj.instanceID);
        [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      end
      if ~tfsucc
        obj.ClearStatus();
        return;
      end
      json = jsondecode(json);
      if ~doblock,
        obj.ClearStatus();
        return;
      end
      
      [tfsucc] = obj.waitForInstanceStart();
      if ~tfsucc,
        warningstr = 'Timed out waiting for AWS EC2 instance to spool up.';
        obj.ClearStatus();
        return;
      end
      
      obj.inspectInstance();
      obj.configureAlarm();
      obj.ClearStatus();
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
      obj.SetStatus('Checking whether AWS EC2 instance is running');
      throwErrs = myparse(varargin,...
        'throwErrs',true... % if false, just warn if there is a problem
        );
      
      if throwErrs
        throwFcn = @error;
      else
        throwFcn = @warningNoTrace;
      end
      
      [tfexist,tfrun] = obj.inspectInstance;
      obj.ClearStatus();

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
      if isalarm,
        return;
      end
      
      codestr = obj.createShutdownAlarmCmd;
      
      fprintf('Setting up AWS CloudWatch alarm to auto-shutdown your instance if it is idle for too long.\n');
      
      tfsucc = obj.syscmd(codestr,...
        'dispcmd',true,...
        'failbehavior','warn');      
    end
    
    function tfsucc = getRemotePythonPID(obj)
      [tfsucc,res] = obj.cmdInstance('pgrep python','dispcmd',true);
      if tfsucc
        pid = str2double(strtrim(res));
        obj.remotePID = pid; % right now each aws instance only has one GPU, so can only do one train/track at a time
        fprintf('Remote PID is: %d.\n\n',pid);
      else
        warningNoTrace('Failed to ascertain remote PID.');
      end
    end
 
    % FUTURE: use rsync if avail. win10 can ask users to setup WSL
    
    function tfsucc = scpDownloadOrVerify(obj,srcAbs,dstAbs,varargin)
      % If dstAbs already exists, does NOT check identity of file against
      % dstAbs. In many cases, naming/immutability of files (with paths)
      % means this is OK.
      
      sysCmdArgs = myparse(varargin,...
        'sysCmdArgs',{});
      
      if exist(dstAbs,'file')>0
        fprintf('File %s exists, not downloading.\n',dstAbs);
        tfsucc = true;
      else
        cmd = AWSec2.scpDownloadCmd(obj.pem,obj.instanceIP,srcAbs,dstAbs,...
          'scpcmd',obj.scpCmd);
        tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
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
        obj.SetStatus(sprintf('Uploading %s file to AWS EC2 instance',fileDescStr));
        fprintf('About to upload. This could take a while depending ...\n');
        tfsucc = obj.scpUpload(src,dstAbs,...
          'destRelative',false,'sysCmdArgs',{'dispcmd',true});
        obj.ClearStatus();
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
      [~,res] = obj.cmdInstance(cmdremote,...
        'dispcmd',dispcmd,'failbehavior','err','usejavaRT',usejavaRT); 
      tf = res(1)=='y';      
    end
    
    function s = remoteFileContents(obj,f,varargin)
      [dispcmd,failbehavior] = myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn'...
        );
      
      cmdremote = sprintf('cat %s',f);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',dispcmd,...
        'failbehavior',failbehavior); 
      if tfsucc  
        s = res;
      else
        % warning thrown etc per failbehavior
        s = '';
      end
    end
    
    function remoteLS(obj,remoteDir,varargin)
      [dispcmd,failbehavior] = myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn'...
        );
      
      cmdremote = sprintf('ls -lh %s',remoteDir);
      [~,res] = obj.cmdInstance(cmdremote,'dispcmd',dispcmd,...
        'failbehavior',failbehavior);
      
      disp(res);
      % warning thrown etc per failbehavior
    end
    
    function remoteDirFull = ensureRemoteDir(obj,remoteDir,varargin)
      % Creates/verifies remote dir. Either succeeds, or fails and harderrors.
      
      [relative,descstr] = myparse(varargin,...
        'relative',true,... true if remoteDir is relative to ~
        'descstr',''... cosmetic, for disp/err strings
        );
      
      if ~isempty(descstr)
        descstr = [descstr ' '];
      end

      if relative
        remoteDirFull = ['~/' remoteDir];
      else
        remoteDirFull = remoteDir;
      end
      
      obj.SetStatus(sprintf('Creating directory %s on AWS EC2 instance',remoteDirFull));
      cmdremote = sprintf('mkdir -p %s',remoteDirFull);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
      obj.ClearStatus();
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
    
    function [tfsucc,res,cmdfull] = cmdInstance(obj,cmdremote,varargin)
      fprintf('cmdInstance: %s\n',cmdremote);
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd,obj.pem,obj.instanceIP,cmdremote,'usedoublequotes',true);
      [tfsucc,res] = AWSec2.syscmd(cmdfull,varargin{:});
    end
        
    function cmd = sshCmdGeneralLogged(obj,cmdremote,logfileremote)
      cmd = AWSec2.sshCmdGeneralLoggedStc(obj.sshCmd,obj.pem,obj.instanceIP,...
        cmdremote,logfileremote);
    end
        
    function tf = canKillRemoteProcess(obj)
      tf = ~isempty(obj.remotePID) && ~isnan(obj.remotePID);
    end
    
    function killRemoteProcess(obj)
      if isempty(obj.remotePID)
        error('Unknown PID for remote process.');
      end
      
      cmdremote = sprintf('kill %d',obj.remotePID);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
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

      assert(iscellstr(fspollargs) && ~isempty(fspollargs));
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
  end
  
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
      cmd = sprintf('aws ec2 describe-instances --filters "Name=image-id,Values=%s" "Name=instance-type,Values=%s" "Name=instance.group-name,Values=%s" "Name=key-name,Values=%s"',ami,instType,secGrp,keyName);
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
      [dispcmd,failbehavior,isjsonout,dosetenv,usejavaRT] = ...
        myparse(varargin,...
        'dispcmd',false,...
        'failbehavior','warn',... % one of 'err','warn','silent'
        'isjsonout',false,...
        'dosetenv',isunix,...
        'usejavaRT',false...
        );
      
%       cmd = [cmd sprintf('\n\r')];
      if dosetenv,
        cmd = [AWSec2.cmdEnv,' ',cmd];
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
          res(1,end+1) = val;
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
    end
    
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
      if ispc 
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

    function cmd = scpDownloadCmd(pem,ip,srcAbs,dstAbs,varargin)
      scpcmd = myparse(varargin,...
        'scpcmd','scp');
      cmd = sprintf('%s -i %s -r ubuntu@%s:"%s" "%s"',scpcmd,pem,ip,srcAbs,dstAbs);
    end

    function cmd = sshCmdGeneral(sshcmd,pem,ip,cmdremote,varargin)
      [timeout,usedoublequotes] = myparse(varargin,...
        'timeout',8,...
        'usedoublequotes',false);
      
      args = {sshcmd '-i' pem sprintf('-oConnectTimeout=%d',timeout) ...
        '-oStrictHostKeyChecking=no' sprintf('ubuntu@%s',ip)};
      if usedoublequotes
        args{end+1} = sprintf('"%s"',cmdremote);
      else
        args{end+1} = sprintf('''%s''',cmdremote);
      end
      cmd = String.cellstr2DelimList(args,' ');
    end

    function cmd = sshCmdGeneralLoggedStc(sshcmd,pem,ip,cmdremote,logfileremote)
      cmd = sprintf('%s -i %s -oStrictHostKeyChecking=no ubuntu@%s "%s </dev/null >%s 2>&1 &"',...
        sshcmd,pem,ip,cmdremote,logfileremote);
    end

    function [tfsucc,instanceID,pemFile] = ...
                              specifyInstanceUIStc(instanceID,pemFile,varargin)
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
    end
    
  end
  
end