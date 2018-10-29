classdef AWSec2 < handle
  
  properties
    instanceID
    instanceIP
    keyName
    pem
    
    scpCmd
    sshCmd
    
    remotePID
    
  end
  
  properties (Constant)
    
    cmdEnv = 'LD_LIBRARY_PATH=: ';
   
  end
  
  methods
    
    function obj = AWSec2(keyName,pem)
      obj.instanceID = [];
      obj.instanceIP = [];
      obj.keyName = keyName;
      obj.pem = pem;
      
      if ispc
        obj.scpCmd = '"c:\Program Files\Git\usr\bin\scp.exe"';
        obj.sshCmd = '"c:\Program Files\Git\usr\bin\ssh.exe"';
      else
        obj.scpCmd = 'scp';
        obj.sshCmd = 'ssh';
      end
      
      obj.remotePID = [];
    end
    
    function delete(obj)
      % TODO 
    end
    
  end
  
  methods
    
    function [tfsucc,json] = launchInstance(obj)
      % sets .instanceID
      
      cmd = AWSec2.launchInstanceCmd(obj.keyName);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        obj.instanceID = [];
        return;
      end
      json = jsondecode(json);
      obj.instanceID = json.Instances.InstanceId;
    end
    
    function [tfsucc,json] = inspectInstance(obj)
       % sets .instanceIP and even .instanceID if it is empty and there is only one instance running
      
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      obj.instanceIP = json.Reservations.Instances.PublicIpAddress;
      if isempty(obj.instanceID)
        obj.instanceID = json.Reservations.Instances.InstanceId;
      else
        assert(strcmp(obj.instanceID,json.Reservations.Instances.InstanceId));
      end
      
      fprintf('EC2 instanceID %s is running with IP %s.\n',obj.instanceID,...
        obj.instanceIP);
    end
    
    function [tfsucc,state,json] = getInstanceState(obj)
      
      state = '';
      
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      state = json.Reservations.Instances.State.Name;
      
    end
    
    function [tfsucc,json] = stopInstance(obj)
      cmd = AWSec2.stopInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
    end

    function [tfsucc,json,warningstr,state] = startInstance(obj,varargin)
      
      [doblock] = myparse(varargin,'doblock',true);
      
      maxwaittime = 100;
      iterwaittime = 5;
      warningstr = '';
      [tfsucc,state,json] = obj.getInstanceState();
      if ~tfsucc,
        warningstr = 'Failed to get instance state.';
        return;
      end
      if ismember(lower(state),{'running','pending'}),
        warningstr = sprintf('Instance is %s, no need to start',state);
        tfsucc = true;
        return;
      end
      if ismember(lower(state),{'shutting-down','terminated'}),
        warningstr = sprintf('Instance is %s, cannot start',state);
        tfsucc = false;
        return
      end
      if ismember(lower(state),{'stopping'}),
        warningstr = sprintf('Instance is %s, please wait for this to finish before starting.',state);
        tfsucc = false;
        return;
      end
      cmd = AWSec2.startInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      if ~doblock,
        return;
      end
      
      % AL: see waitforPoll() util
      starttime = tic;
      tfsucc = false;
      while true,
        [tf,state1] = obj.getInstanceState();
        if tf && strcmpi(state1,'running'),
          tfsucc = true;
          break;
        end
        if toc(starttime) > maxwaittime,
          return;
        end
        pause(iterwaittime);
      end
      obj.inspectInstance();
    end

    
    function checkInstanceRunning(obj)
      % - If runs silently, obj appears to be a running EC2 instance with no
      %   issues
      % - If harderror thrown, something appears wrong
      
      [tf,js] = obj.inspectInstance;
      if ~tf
        error('Problem with EC2 instance id: %s',obj.instanceID);
      end
      state = js.Reservations.Instances.State;
      if ~strcmp(state.Name,'running')
        error('EC2 instance id %s is not in the ''running'' state.',...
          obj.instanceID)
      end
    end
 
    function tfsucc = scpDownload(obj,srcAbs,dstAbs,varargin)
      % overwrites with no regard for anything
      
      sysCmdArgs = myparse(varargin,...
        'sysCmdArgs',{});
      cmd = AWSec2.scpDownloadCmd(obj.pem,obj.instanceIP,srcAbs,dstAbs,...
        'scpcmd',obj.scpCmd);
      tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
    end
 
    function tfsucc = scpUpload(obj,file,dstRel,varargin)
      sysCmdArgs = myparse(varargin,...
        'sysCmdArgs',{});
      cmd = AWSec2.scpUploadCmd(file,obj.pem,obj.instanceIP,dstRel,...
        'scpcmd',obj.scpCmd);
      tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
    end
    
    function scpUploadOrVerify(obj,src,dstRel,fileDescStr) % throws
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
            
      tfsucc = obj.remoteFileExists(dstRel,'dispcmd',true);
      if tfsucc
        fprintf('%s file exists: %s.\n\n',...
          String.niceUpperCase(fileDescStr),dstRel);
      else
        fprintf('About to upload. This could take a while depending ...\n');
        tfsucc = obj.scpUpload(src,dstRel,'sysCmdArgs',{'dispcmd',true});
        if tfsucc
          fprintf('Uploaded %s %s to %s.\n\n',fileDescStr,src,dstRel);
        else
          error('Failed to upload %s %s.',fileDescStr,src);
        end
      end
    end
    
    function [tfsucc,res,cmdfull] = cmdInstance(obj,cmdremote,varargin)
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd,obj.pem,obj.instanceIP,cmdremote);
      [tfsucc,res] = AWSec2.syscmd(cmdfull,varargin{:});
    end
        
    function cmd = sshCmdGeneralLogged(obj,cmdremote,logfileremote)
      cmd = AWSec2.sshCmdGeneralLoggedStc(obj.sshCmd,obj.pem,obj.instanceIP,...
        cmdremote,logfileremote);
    end
    
    function tf = remoteFileExists(obj,f,varargin)
      [reqnonempty,dispcmd,usejavaRT] = myparse(varargin,...
        'reqnonempty',false,...
        'dispcmd',false,...
        'usejavaRT',false...
        );

      if reqnonempty
        script = '~/APT/misc/fileexistsnonempty.sh';
      else
        script = '~/APT/misc/fileexists.sh';
      end
      cmdremote = sprintf('%s %s',script,f);
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
        
  end
  
  methods (Static)
    
    function cmd = launchInstanceCmd(keyName,varargin)
      [ami,instType,secGrp] = myparse(varargin,...
        'ami','ami-0168f57fb900185e1',...
        'instType','p3.2xlarge',...
        'secGrp','apt_dl');
      cmd = sprintf('aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s --key-name %s',ami,instType,secGrp,keyName);
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
        [st,res] = system(cmd);
        tfsucc = st==0;
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
    
    function cmd = scpUploadCmd(file,pem,ip,dstrel,varargin)
      scpcmd = myparse(varargin,...
        'scpcmd','scp');
      cmd = sprintf('%s -i %s %s ubuntu@%s:~/%s',scpcmd,pem,file,ip,dstrel);
    end

    function cmd = scpDownloadCmd(pem,ip,srcAbs,dstAbs,varargin)
      scpcmd = myparse(varargin,...
        'scpcmd','scp');
      cmd = sprintf('%s -i %s -r ubuntu@%s:%s %s',scpcmd,pem,ip,srcAbs,dstAbs);
    end

    function cmd = sshCmdGeneral(sshcmd,pem,ip,cmdremote)
      cmd = sprintf('%s -i %s -oStrictHostKeyChecking=no ubuntu@%s ''%s''',sshcmd,pem,ip,cmdremote);
    end

    function cmd = sshCmdGeneralLoggedStc(sshcmd,pem,ip,cmdremote,logfileremote)
      cmd = sprintf('%s -i %s -oStrictHostKeyChecking=no ubuntu@%s "%s </dev/null >%s 2>&1 &"',...
        sshcmd,pem,ip,cmdremote,logfileremote);
    end

  end
  
end