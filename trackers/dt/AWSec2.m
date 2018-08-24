classdef AWSec2 < handle
  
  properties
    instanceID
    instanceIP
    keyName
    pem
  end
  
  methods
    
    function obj = AWSec2(keyName,pem)
      obj.instanceID = [];
      obj.instanceIP = [];
      obj.keyName = keyName;
      obj.pem = pem;
    end
    
    function [tfsucc,json] = launchInstance(obj)
      % sets .instanceID
      
      cmd = obj.launchInstanceCmd(obj.keyName);
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true);
      if ~tfsucc
        obj.instanceID = [];
        return;
      end
      json = jsondecode(json);
      obj.instanceID = json.InstanceId;
    end
    
    function [tfsucc,json] = inspectInstance(obj)
      % sets .instanceIP and even .instanceID in some cases
      
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID
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
    
    function tfsucc = scpUpload(obj,file)
      cmd = AWSec2.scpFileCmd(file,obj.pem,obj.instanceIP);
      tfsucc = AWSec2.syscmd(cmd,'dispcmd',true);
    end
    
    function [tfsucc,res,cmdfull] = cmdInstance(obj,cmdremote,varargin)
      cmdfull = AWSec2.sshCmdGeneral(obj.pem,obj.instanceIP,cmdremote);
      [tfsucc,res] = AWSec2.syscmd(cmdfull,varargin{:});
    end
        
  end
  
  methods (Static)
    
    function [tfsucc,res] = syscmd(cmd,varargin)
      [dispcmd,harderronfail] = myparse(varargin,...
        'dispcmd',false,...
        'harderronfail',false...
        );
      
      cmd = [cmd sprintf('\n\r')];
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
        'ami','ami-0a5758d56c4c40ae6',...
        'instType','p2.xlarge',...
        'secGrp','apt_dl');
      cmd = sprintf('aws ec2 run-instances --image-id %s --count 1 --instance-type %s --security-groups %s --key-name %s',ami,instType,secGrp,keyName);
    end
    
    function cmd = describeInstancesCmd(ec2id)
      cmd = sprintf('aws ec2 describe-instances --instance-ids %s',ec2id);      
    end
    
    function cmd = scpFileCmd(file,pem,ip)
      cmd = sprintf('scp -i %s %s ubuntu@%s:~',pem,file,ip);
    end
    
    function cmd = sshCmdGeneral(pem,ip,cmdremote)
      cmd = sprintf('ssh -i %s ubuntu@%s "%s"',pem,ip,cmdremote);
    end
    
    function cmdremote = sshCmdTrain(pem,ip,lblS,cacheS,trnID,view0based)      
      cmdremote = {
        'cd /home/ubuntu/APT/deepnet;'; 
        'git pull;'; 
        'git checkout feature/deeptrack;';
        'LD_LIBRARY_PATH=/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib;';
        sprintf('python APT_interface.py -cache /home/ubuntu/%s -name %s -view %d /home/ubuntu/%s train -use_cache;',...
          cacheS,trnID,view0based,lblS);
        };
      cmdremote = cat(2,cmdremote{:});
    end

  end
  
end