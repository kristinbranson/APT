classdef AWSec2 < handle
  
  properties
    instanceID
    instanceIP
    keyName
    pem
    
    scpCmd = '"c:\Program Files\Git\usr\bin\scp.exe"';
    sshCmd = '"c:\Program Files\Git\usr\bin\ssh.exe"';
  end
  
  methods
    
    function obj = AWSec2(keyName,pem)
      obj.instanceID = [];
      obj.instanceIP = [];
      obj.keyName = keyName;
      obj.pem = pem;
    end
    
    function delete(obj)
      % TODO 
    end
  end
  
  methods
    
    function [tfsucc,json] = launchInstance(obj)
      % sets .instanceID
      
      cmd = obj.launchInstanceCmd(obj.keyName);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true);
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
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true);
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
    end
    
    function [tfsucc,json] = stopInstance(obj)
      cmd = AWSec2.stopInstanceCmd(obj.instanceID);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true);
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
    
    function tfsucc = scpUpload(obj,file,dstRel,varargin)
      [sysCmdArgs] = myparse(varargin,...
        'sysCmdArgs',{});
      cmd = AWSec2.scpFileCmd(file,obj.pem,obj.instanceIP,dstRel,...
        'scpcmd',obj.scpCmd);
      tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
    end
    
    function [tfsucc,res,cmdfull] = cmdInstance(obj,cmdremote,varargin)
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd,obj.pem,obj.instanceIP,cmdremote);
      [tfsucc,res] = AWSec2.syscmd(cmdfull,varargin{:});
    end
        
    function cmd = sshCmdGeneralLogged(obj,cmdremote,logfileremote)
      cmd = AWSec2.sshCmdGeneralLoggedStc(obj.sshCmd,obj.pem,obj.instanceIP,...
        cmdremote,logfileremote);
    end

  end
  
  methods (Static)
    
    function [tfsucc,res] = syscmd(cmd,varargin)
      [dispcmd,harderronfail] = myparse(varargin,...
        'dispcmd',false,...
        'harderronfail',false...
        );
      
%       cmd = [cmd sprintf('\n\r')];
      if dispcmd
        disp(cmd); 
      end
      [st,res] = system(cmd);
      tfsucc = st==0;
      if ~tfsucc 
        if harderronfail
          error('Nonzero status code: %s',res);
        else
          warningNoTrace('Command failed: %s: %s',cmd,res);
        end
      end
    end
    
    function cmd = launchInstanceCmd(keyName,varargin)
      [ami,instType,secGrp] = myparse(varargin,...
        'ami','ami-0168f57fb900185e1',...
        'instType','p2.xlarge',...
        'secGrp','apt_dl');
      cmd = sprintf('aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s --key-name %s',ami,instType,secGrp,keyName);
    end
    
    function cmd = describeInstancesCmd(ec2id)
      cmd = sprintf('aws ec2 describe-instances --instance-ids %s',ec2id);      
    end
    
    function cmd = stopInstanceCmd(ec2id)
      cmd = sprintf('aws ec2 stop-instances --instance-ids %s',ec2id);
    end
    
    function cmd = scpFileCmd(file,pem,ip,dstrel,varargin)
      scpcmd = myparse(varargin,...
        'scpcmd','scp');
      cmd = sprintf('%s -i %s %s ubuntu@%s:~/%s',scpcmd,pem,file,ip,dstrel);
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