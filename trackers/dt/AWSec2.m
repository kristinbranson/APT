classdef AWSec2 < handle
  
  properties
    instanceID
    instanceIP
    keyName
    pem
    
    scpCmd
    sshCmd
    
    remotePID
    
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
      
      cmd = obj.launchInstanceCmd(obj.keyName);
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
      
      cmd = obj.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
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
    
    function [tfsucc,json] = stopInstance(obj)
      cmd = obj.stopInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
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
      [reqnonempty,dispcmd] = myparse(varargin,...
        'reqnonempty',false,...
        'dispcmd',false...
        );

      if reqnonempty
        script = '~/APT/misc/fileexistsnonempty.sh';
      else
        script = '~/APT/misc/fileexists.sh';
      end
      cmdremote = sprintf('%s %s',script,f);
      [~,res] = obj.cmdInstance(cmdremote,...
        'dispcmd',dispcmd,'harderronfail',true); 
      tf = res(1)=='y';      
    end
    
    function s = remoteFileContents(obj,f,varargin)
      [dispcmd,harderronfail] = myparse(varargin,...
        'dispcmd',false,...
        'harderronfail',false);
      
      cmdremote = sprintf('cat %s',f);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',dispcmd,'harderronfail',harderronfail); 
      if tfsucc  
        s = res;
      else
        % harderronfail==false, warning thrown
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
    
    function cmd = launchInstanceCmd(obj,keyName,varargin)
      [ami,instType,secGrp] = myparse(varargin,...
        'ami','ami-0168f57fb900185e1',...
        'instType','p2.xlarge',...
        'secGrp','apt_dl');
      cmd = sprintf('%s aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s --key-name %s',obj.cmdEnv,ami,instType,secGrp,keyName);
    end
    
    function cmd = describeInstancesCmd(obj,ec2id)
      cmd = sprintf('%s aws ec2 describe-instances --instance-ids %s',obj.cmdEnv,ec2id);
    end
    
    function cmd = stopInstanceCmd(obj,ec2id)
      cmd = sprintf('%s aws ec2 stop-instances --instance-ids %s',obj.cmdEnv,ec2id);
    end
    
  end
  
  methods (Static)
    
    function [tfsucc,res,warningstr] = syscmd(cmd,varargin)
      [dispcmd,harderronfail,isjsonout] = myparse(varargin,...
        'dispcmd',false,...
        'harderronfail',false,...
        'isjsonout',false...
        );
      
%       cmd = [cmd sprintf('\n\r')];
      if dispcmd
        disp(cmd); 
      end
      [st,res] = system(cmd);
      tfsucc = st==0;
      
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
        if harderronfail
          error('Nonzero status code: %s',res);
        else
          warningNoTrace('Command failed: %s: %s',cmd,res);
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