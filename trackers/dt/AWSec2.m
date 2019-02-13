classdef AWSec2 < handle
  % Handle to a single AWS EC2 instance. The instance may be in any state,
  % running, stopped, etc.
  
  
  properties
    instanceID % primary ID. depending on config, IPs can change when instances are stopped/restarted etc.
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
    
    function obj = AWSec2(pem,varargin)
      obj.pem = pem;
      
      if ispc
        obj.scpCmd = '"c:\Program Files\Git\usr\bin\scp.exe"';
        obj.sshCmd = '"c:\Program Files\Git\usr\bin\ssh.exe"';
      else
        obj.scpCmd = 'scp';
        obj.sshCmd = 'ssh';
      end

      for i=1:2:numel(varargin)
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
    
    function [tfexist,tfrunning,json] = inspectInstance(obj)
      % Check that a specified instance exists; check if it is running; 
      % get json; sets .instanceIP if running
      %
      % * If .instanceID is empty and there is a unique EC2 instance, this 
      % will set the .instanceID.
      % 
      % * tfexist is returned as true of the instance exists in some state.
      % * tfrunning is returned as true if the instance exists and is running.
      % * json is valid only if tfexist==true.
      
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
      [tfexist,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfexist
        tfrunning = false;
        return;
      end
      json = jsondecode(json);
      
      inst = json.Reservations.Instances;
      if isempty(obj.instanceID)
        obj.instanceID = inst.InstanceId;
      else
        assert(strcmp(obj.instanceID,inst.InstanceId));
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
      
      state = '';
      
      cmd = AWSec2.describeInstancesCmd(obj.instanceID); % works with empty .instanceID if there is only one instance
      [tfsucc,json] = AWSec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      if ~tfsucc
        return;
      end
      json = jsondecode(json);
      state = json.Reservations.Instances.State.Name;
      
    end
    
    function tfsucc = respecifyInstance(obj)
      [tfsucc,iid,pemFile] = ...
        obj.specifyInstanceUIStc(obj.instanceID,obj.pem);
      if tfsucc
        obj.instanceID = iid;
        obj.pem = pemFile;
      end
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
    
    function checkInstanceRunning(obj,varargin)
      % - If runs silently, obj appears to be a running EC2 instance with 
      %   no issues
      % - If harderror thrown, something appears wrong
      
      throwErrs = myparse(varargin,...
        'throwErrs',true... % if false, just warn if there is a problem
        );
      
      if throwErrs
        throwFcn = @error;
      else
        throwFcn = @warningNoTrace;
      end
      
      [tfexist,tfrun] = obj.inspectInstance;
      if ~tfexist
        throwFcn('Problem with EC2 instance id: %s',obj.instanceID);
      end
      if ~tfrun
        throwFcn('EC2 instance id %s is not in the ''running'' state.',...
          obj.instanceID)
      end
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
 
    function tfsucc = scpDownload(obj,srcAbs,dstAbs,varargin)
      % overwrites with no regard for anything
      
      sysCmdArgs = myparse(varargin,...
        'sysCmdArgs',{});
      cmd = AWSec2.scpDownloadCmd(obj.pem,obj.instanceIP,srcAbs,dstAbs,...
        'scpcmd',obj.scpCmd);
      tfsucc = AWSec2.syscmd(cmd,sysCmdArgs{:});
    end
    
    function tfsucc = scpDownloadEnsureDir(obj,srcAbs,dstAbs,varargin)
      dirLcl = fileparts(dstAbs);
      if exist(dirLcl,'dir')==0
        [tfsucc,msg] = mkdir(dirLcl);
        if ~tfsucc
          warningNoTrace('Failed to create local directory %s: %s',dirLcl,msg);
          return;
        end
      end
      tfsucc = obj.scpDownload(srcAbs,dstAbs,varargin{:});
    end
 
    function tfsucc = scpUpload(obj,file,dest,varargin)
      [destRelative,sysCmdArgs] = myparse(varargin,...
        'destRelative',true,... % true if dest is relative to ~
        'sysCmdArgs',{});
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
        fprintf('About to upload. This could take a while depending ...\n');
        tfsucc = obj.scpUpload(src,dstAbs,...
          'destRelative',false,'sysCmdArgs',{'dispcmd',true});
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
        script = '~/APT/misc/fileexistsnonempty.sh';
      else
        script = '~/APT/misc/fileexists.sh';
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
      
      cmdremote = sprintf('mkdir -p %s',remoteDirFull);
      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
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
      cmdfull = AWSec2.sshCmdGeneral(obj.sshCmd,obj.pem,obj.instanceIP,cmdremote,'usedoublequotes',true);
      [tfsucc,res] = AWSec2.syscmd(cmdfull,varargin{:});
    end
        
    function cmd = sshCmdGeneralLogged(obj,cmdremote,logfileremote)
      cmd = AWSec2.sshCmdGeneralLoggedStc(obj.sshCmd,obj.pem,obj.instanceIP,...
        cmdremote,logfileremote);
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
      cmdremote = sprintf('~/APT/misc/fspoll.py %s',fspollstr);

      [tfsucc,res] = obj.cmdInstance(cmdremote,'dispcmd',true);
      if tfsucc
        res = regexp(res,'\n','split');
        tfsucc = iscell(res) && numel(res)==nresps+1; % last cell is {0x0 char}
        res = res(1:end-1);
      else
        res = [];
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
    
    function cmd = scpUploadCmd(file,pem,ip,dest,varargin)
      [destRelative,scpcmd] = myparse(varargin,...
        'destRelative',true,...
        'scpcmd','scp');
      if destRelative
        dest = ['~/' dest];
      end
      cmd = sprintf('%s -i %s %s ubuntu@%s:%s',scpcmd,pem,file,ip,dest);
    end

    function cmd = scpDownloadCmd(pem,ip,srcAbs,dstAbs,varargin)
      scpcmd = myparse(varargin,...
        'scpcmd','scp');
      cmd = sprintf('%s -i %s -r ubuntu@%s:%s %s',scpcmd,pem,ip,srcAbs,dstAbs);
    end

    function cmd = sshCmdGeneral(sshcmd,pem,ip,cmdremote,varargin)
      [timeout,usedoublequotes] = myparse(varargin,...
        'timeout',5,...
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
    
  end
  
end