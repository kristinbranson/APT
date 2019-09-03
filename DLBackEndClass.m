classdef DLBackEndClass < handle
  % Design unclear but good chance this is a thing
  %
  % This thing (maybe) specifies a physical machine/server along with a 
  % DLBackEnd:
  % * DLNetType: what DL net is run
  % * DLBackEndClass: where/how DL was run
  %
  % TODO: this should be named 'DLBackEnd' and 'DLBackEnd' should be called
  % 'DLBackEndType' or something.
  
  properties (Constant)
    minFreeMem = 9000; % in MiB
  end
  properties
    type  % scalar DLBackEnd
    
    % scalar logical. if true, backend runs code in APT.Root/deepnet. This
    % path must be visible in the backend or else.
    %
    % Conceptually this could be an arbitrary loc.
    deepnetrunlocal = true; 
    
    awsec2 % used only for type==AWS
    
    aptdockerimgroot = 'bransonlabapt/apt_docker';
    aptdockerimgtag = ''; % optional tag eg 'tf1.6'
    
    condaEnv = 'APT'; % used only for Conda
  end
  properties (Dependent)
    filesep
    aptdockerimg % full docker img spec (with tag if specified)
  end
  
  methods
    function v = get.filesep(obj)
      if obj.type == DLBackEnd.Conda,
        v = filesep;
      else
        v = '/';
      end
    end
    function v = get.aptdockerimg(obj)
      v = obj.aptdockerimgroot;
      if ~isempty(obj.aptdockerimgtag)
        v = [v ':' obj.aptdockerimgtag];
      end
    end
  end
 
  methods
    
    function obj = DLBackEndClass(ty,oldbe)
      if nargin > 1,
        % save state
        obj.deepnetrunlocal = oldbe.deepnetrunlocal;
        obj.awsec2 = oldbe.awsec2;
        obj.aptdockerimgroot = oldbe.aptdockerimgroot;
        obj.aptdockerimgtag = oldbe.aptdockerimgtag;
        obj.condaEnv = oldbe.condaEnv;
      end
      
      obj.type = ty;
    end
    
    function delete(obj)
      if obj.type==DLBackEnd.AWS
        aws = obj.awsec2;
        if ~isempty(aws)
          fprintf(1,'Stopping AWS EC2 instance %s.',aws.instanceID);
          tfsucc = aws.stopInstance();
          if ~tfsucc
            warningNoTrace('Failed to stop AWS EC2 instance %s.',aws.instanceID);
          end
        end
      end
    end
    
    function testConfigUI(obj,cacheDir)
      % Test whether backend is ready to do; display results in msgbox
      
      switch obj.type,
        case DLBackEnd.Bsub,
          DLBackEndClass.testBsubConfig(cacheDir);
        case DLBackEnd.Docker
          obj.testDockerConfig();
        case DLBackEnd.AWS
          obj.testAWSConfig();
        otherwise
          msgbox(sprintf('Tests for %s have not been implemented',obj.type),...
            'Not implemented','modal');
      end
    end
    
    function [tf,reason] = getReadyTrainTrack(obj)
      tf = false;
      if obj.type==DLBackEnd.AWS
        aws = obj.awsec2;
        
        didLaunch = false;
        if ~obj.awsec2.isConfigured || ~obj.awsec2.isSpecified,
          [tfsucc,instanceID,instanceType,reason,didLaunch] = ...
            obj.awsec2.selectInstance(...
            'canconfigure',1,'canlaunch',1,'forceselect',0);
          if ~tfsucc || isempty(instanceID),
            reason = sprintf('Problem configuring: %s',reason);
            return;
          end
        end

        
        [tfexist,tfrunning] = obj.awsec2.inspectInstance;
        if ~tfexist,
          uiwait(warndlg(sprintf('AWS EC2 instance %s could not be found or is terminated. Please configure AWS back end with a different AWS EC2 instance.',obj.awsec2.instanceID),'AWS EC2 instance not found'));
          reason = 'Instance could not be found.';
          obj.awsec2.ResetInstanceID();
          return;
        end
        
        tf = tfrunning;
        if ~tf
          if didLaunch,
            btn = 'Yes';
          else
            qstr = sprintf('AWS EC2 instance %s is not running. Start it?',obj.awsec2.instanceID);
            tstr = 'Start AWS EC2 instance';
            btn = questdlg(qstr,tstr,'Yes','Cancel','Cancel');
            if isempty(btn)
              btn = 'Cancel';
            end
          end
          switch btn
            case 'Yes'
              tf = obj.awsec2.startInstance();
              if ~tf
                reason = sprintf('Could not start AWS EC2 instance %s.',obj.awsec2.instanceID);
                return;
              end
            otherwise
              reason = sprintf('AWS EC2 instance %s is not running.',obj.awsec2.instanceID);
              return;
          end
        end
        
        reason = '';
      else
        tf = true;
        reason = '';
      end
    end
    
    function s = prettyName(obj)
      switch obj.type,
        case DLBackEnd.Bsub,
          s = 'JRC Cluster';
        case DLBackEnd.Docker,
          s = 'Local';
        otherwise
          s = char(obj.type);
      end
    end
    
    function [gpuid,freemem,gpuInfo] = getFreeGPUs(obj,nrequest,varargin)
      
      [dockerimg,minFreeMem,condaEnv,verbose] = myparse(varargin,...
        'dockerimg',obj.aptdockerimg,...
        'minfreemem',obj.minFreeMem,...
        'condaEnv',obj.condaEnv,...
        'verbose',0 ...
      ); %#ok<PROPLC>
      
      gpuid = [];
      freemem = 0;
      gpuInfo = [];
      aptdeepnet = APT.getpathdl;
      
      switch obj.type,
        case DLBackEnd.Docker,
          if isempty(APT.DOCKER_REMOTE_HOST),
            dockercmd = 'docker';
            dockercmdend = '';
            filequote = '"';
          else
            dockercmd = sprintf('ssh -t %s "docker',APT.DOCKER_REMOTE_HOST);
            dockercmdend = '"';
            filequote = '\"';
          end
    
          parsenvidiasmi = sprintf('%s%sparse_nvidia_smi.py',aptdeepnet,obj.filesep);
          parsenvidiasmi = [filequote parsenvidiasmi filequote];
          aptdeepnetguard = [filequote aptdeepnet filequote];
          %aptdeepnetguard = String.escapeSpaces(aptdeepnet);
          basecmd = sprintf('echo START; python %s; echo END',parsenvidiasmi);
          bindpath = {aptdeepnet}; % don't use guarded
          mountArgs = cellfun(@(x)sprintf('--mount ''type=bind,src=%s,dst=%s''',x,x),bindpath,'uni',0);
          mountArgs = sprintf('%s ',mountArgs{:});
          codestr = {...
            dockercmd 'run' '-i' ...
            '--runtime' 'nvidia' ...
            '--rm' ...
            '--user $(id -u)' ...
            '-w' aptdeepnetguard ...
            mountArgs ...
            dockerimg ...
            sprintf('bash -c ''cd %s; %s''%s',aptdeepnetguard,basecmd,dockercmdend) ...
            };
          codestr = String.cellstr2DelimList(codestr,' ');
          %codestr = sprintf('%s run -i --runtime nvidia --rm --user $(id -u) -w %s %s %s bash -c ''cd %s; %s''%s',dockercmd,aptdeepnet,mountArgs,dockerimg,aptdeepnet,basecmd,dockercmdend);
          if verbose
            fprintf(1,'%s\n',codestr);
          end
          [st,res] = system(codestr);
          if st ~= 0,
            warning('Error getting GPU info: %s',res);
            return;
          end
        case DLBackEnd.Conda
          basecmd = sprintf('echo START && python %s%sparse_nvidia_smi.py && echo END',...
            aptdeepnet,obj.filesep);
          codestr = sprintf('activate %s && %s',condaEnv,basecmd);
          [st,res] = system(codestr);
          if st ~= 0,
            warning('Error getting GPU info: %s',res);
            return;
          end
        otherwise
          error('Not implemented');
      end
      
      res0 = res;
      res = regexp(res,'\n','split');
      res = strip(res);
      i0 = find(strcmp(res,'START'),1);
      if isempty(i0),
        warning('Could not find START of GPU info');
        disp(res0);
        return;
      end
      i0 = i0+1;
      i1 = find(strcmp(res(i0+1:end),'END'),1)+i0;
      res = res(i0+1:i1-1);
      ngpus = numel(res);      
      gpuInfo = struct;
      gpuInfo.id = zeros(1,ngpus);
      gpuInfo.totalmem = zeros(1,ngpus);
      gpuInfo.freemem = zeros(1,ngpus);
      for i = 1:ngpus,
        v = str2double(strsplit(res{i},','));
        gpuInfo.id(i) = v(1);
        gpuInfo.freemem(i) = v(2);
        gpuInfo.totalmem(i) = v(3);
      end      
      
      [freemem,order] = sort(gpuInfo.freemem,'descend');
      if freemem(min(nrequest,numel(freemem))) < minFreeMem, %#ok<PROPLC>
        i = find(freemem>=minFreeMem,1,'last'); %#ok<PROPLC>
        freemem = freemem(1:i);
        gpuid = gpuInfo.id(order(1:i));
        return;
      end
      freemem = freemem(1:nrequest);
      gpuid = gpuInfo.id(order(1:nrequest));        
    end
    
%     function tf = filesysAreCompatible(obj,obj2)
%       assert(isscalar(obj) && isscalar(obj2));
%     end
  end
  
   methods (Static)    
    
    function [hfig,hedit] = createFigTestConfig(figname)
      hfig = dialog('Name',figname,'Color',[0,0,0],'WindowStyle','normal');
      hedit = uicontrol(hfig,'Style','edit','Units','normalized',...
      'Position',[.05,.05,.9,.9],'Enable','inactive','Min',0,'Max',10,...
      'HorizontalAlignment','left','BackgroundColor',[.1,.1,.1],...
      'ForegroundColor',[0,1,0]);
    end
    
    function [tfsucc,hedit] = testBsubConfig(cacheDir,varargin)
      tfsucc = false;
      [host] = myparse(varargin,'host',DeepTracker.jrchost);

      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test JRC Cluster Backend');      
      hedit.String = {sprintf('%s: Testing JRC cluster backend...',datestr(now))}; 
      drawnow;
      
      % test that you can ping jrc host
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing that host %s can be reached...\n',host); drawnow;
      cmd = sprintf('ping -c 1 -W 10 %s',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = 'FAILURE. Error with ping command.'; drawnow;
        return;
      end
      m = regexp(result,' (\d+) received, (\d+)% packet loss','tokens','once');
      if isempty(m),
        hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
        return;
      end
      if str2double(m{1}) == 0,
        hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
      % test that we can connect to jrc host and access CacheDir on it
     
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
      touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));
      
      remotecmd = sprintf('touch "%s"; if [ -e "%s" ]; then rm -f "%s" && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
      cmd1 = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host,'bg',false);
      cmd = sprintf('timeout 20 %s',cmd1);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); drawnow;
        return;
      end
      issuccess = contains(result,'SUCCESS');
      isfailure = contains(result,'FAILURE');
      if issuccess && ~isfailure,
        hedit.String{end+1} = 'SUCCESS!'; drawnow;
      elseif ~issuccess && isfailure,
        hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
        return;
      else
        hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
        return;
      end
      
      % test that we can run bjobs
      hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
      remotecmd = 'bjobs';
      cmd = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; 
      hedit.String{end+1} = ''; 
      hedit.String{end+1} = 'All tests passed. JRC Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end    
   end
   
   methods

    function [tfsucc,hedit] = testDockerConfig(obj)
      tfsucc = false;
      %[host] = myparse(varargin,'host',DeepTracker.jrchost);

      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test Docker Configuration');      
      hedit.String = {sprintf('%s: Testing Docker Configuration...',datestr(now))}; 
      drawnow;
      
      % docker hello world
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing docker hello-world...'; drawnow;
      
      if isempty(APT.DOCKER_REMOTE_HOST),
        dockercmd = 'docker';
        dockercmdend = '';
        filequote = '"';
      else
        dockercmd = sprintf('ssh -t %s "docker',APT.DOCKER_REMOTE_HOST);
        dockercmdend = '"';
        filequote = '\"';
      end      
      cmd = sprintf('%s run hello-world%s',dockercmd,dockercmdend);
      fprintf(1,'%s\n',cmd);
      hedit.String{end+1} = cmd; drawnow;
      [st,res] = system(cmd);
      reslines = splitlines(res);
      reslinesdisp = reslines(1:min(4,end));
      hedit.String = [hedit.String; reslinesdisp(:)];
      if st~=0
        hedit.String{end+1} = 'FAILURE. Error with docker run command.'; drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
%       % nvidia-docker nvidia-smi
%       hedit.String{end+1} = ''; drawnow;
%       hedit.String{end+1} = '** Testing nvidia-docker with nvidia-smi...\n'; drawnow;
%       cmd = 'docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi';
%       hedit.String{end+1} = cmd; drawnow;
%       [st,res] = system(cmd);
%       reslines = splitlines(res);
%       reslinesdisp = reslines(1:min(4,end));
%       hedit.String = [hedit.String; reslinesdisp(:)];
%       if st~=0
%         hedit.String{end+1} = 'FAILURE. Error with nvidia-docker run command.'; drawnow;
%         return;
%       end
%       hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
      % APT hello
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing APT deepnet library...'; drawnow;
      deepnetroot = [APT.Root '/deepnet'];
      deepnetrootguard = [filequote deepnetroot filequote];
%       if isempty(APT.DOCKER_REMOTE_HOST),
%         dockercmd = 'docker';
%         dockercmdend = '';
%       else
%         dockercmd = sprintf('ssh -t %s "docker',APT.DOCKER_REMOTE_HOST);
%         dockercmdend = '"';
%       end
      cmd = { ...
        dockercmd 'run' '-it' ...
        '--runtime' 'nvidia' ...
        '--rm' ...
        sprintf('--mount ''type=bind,src=%s,dst=%s''',deepnetroot,deepnetroot) ...
        '-w' deepnetrootguard ...
        obj.aptdockerimg ...
        sprintf('python APT_interface.py lbl test hello%s',dockercmdend) ...
        };
      cmd = String.cellstr2DelimList(cmd,' ');
%       cmd = sprintf('%s run -it --runtime nvidia --rm -v %s:%s -w %s %s python APT_interface.py lbl test hello%s',...
%         dockercmd,deepnetroot,deepnetroot,deepnetroot,obj.aptdockerimg,dockercmdend);
      RUNAPTHELLO = 1;
      if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
        fprintf(1,'%s\n',cmd);
        hedit.String{end+1} = cmd; drawnow;
        [st,res] = system(cmd);
        reslines = splitlines(res);
        reslinesdisp = reslines(1:min(4,end));
        hedit.String = [hedit.String; reslinesdisp(:)];
        if st~=0
          hedit.String{end+1} = 'FAILURE. Error with APT deepnet command.'; drawnow;
          return;
        end
        hedit.String{end+1} = 'SUCCESS!'; drawnow;
      end
      
      % free GPUs
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Looking for free GPUs ...\n'; drawnow;
      [gpuid,freemem,gpuifo] = obj.getFreeGPUs(1,'verbose',true);
      if isempty(gpuid)
        hedit.String{end+1} = 'FAILURE. Could not find free GPUs.'; drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS! Found available GPUs.'; drawnow;
      
      hedit.String{end+1} = '';
      hedit.String{end+1} = 'All tests passed. Docker Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end
    
    function [tfsucc,hedit] = testAWSConfig(obj,varargin)
      tfsucc = false;
      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test AWS Backend');
      hedit.String = {sprintf('%s: Testing AWS backend...',datestr(now))}; 
      drawnow;
      
      % test that ssh exists
      hedit.String{end+1} = sprintf('** Testing that ssh is available...'); drawnow;
      hedit.String{end+1} = ''; drawnow;
      if ispc,
        isssh = exist(APT.WINSSHCMD,'file') && exist(APT.WINSCPCMD,'file');
        if isssh,
          hedit.String{end+1} = sprintf('Found ssh at %s',APT.WINSSHCMD); 
          drawnow;
        else
          hedit.String{end+1} = sprintf('FAILURE. Did not find ssh in the expected location: %s.',APT.WINSSHCMD); 
          drawnow;
          return;
        end
      else
        cmd = 'which ssh';
        hedit.String{end+1} = cmd; drawnow;
        [status,result] = system(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find ssh.'; drawnow;
          return;
        end
      end
      
      if ispc,
        hedit.String{end+1} = sprintf('\n** Testing that certUtil is installed...\n'); drawnow;
        cmd = 'where certUtil';
        hedit.String{end+1} = cmd; drawnow;
        [status,result] = system(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find certUtil.'; drawnow;
          return;
        end
      end

      % test that AWS CLI is installed
      hedit.String{end+1} = sprintf('\n** Testing that AWS CLI is installed...\n'); drawnow;
      cmd = 'aws ec2 describe-regions --output table';
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = 'FAILURE. Error using the AWS CLI.'; drawnow;
        return;
      end

      % test that apt_dl security group has been created
      hedit.String{end+1} = sprintf('\n** Testing that apt_dl security group has been created...\n'); drawnow;
      cmd = 'aws ec2 describe-security-groups';
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      if status == 0,
        try
          result = jsondecode(result);
          if ismember('apt_dl',{result.SecurityGroups.GroupName}),
            hedit.String{end+1} = 'Found apt_dl security group.'; drawnow;
          else
            status = 1;
          end
        catch
          status = 1;
        end
        if status == 1,
          hedit.String{end+1} = 'FAILURE. Could not find the apt_dl security group.'; drawnow;
        end
      else
        hedit.String{end+1} = result; drawnow;
        hedit.String{end+1} = 'FAILURE. Error checking for apt_dl security group.'; drawnow;
        return;
      end
      
      % to do, could test launching an instance, or at least dry run

%       m = regexp(result,' (\d+) received, (\d+)% packet loss','tokens','once');
%       if isempty(m),
%         hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
%         return;
%       end
%       if str2double(m{1}) == 0,
%         hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
%         return;
%       end
%       hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       
%       % test that we can connect to jrc host and access CacheDir on it
%      
%       hedit.String{end+1} = ''; drawnow;
%       hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
%       touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));
%       
%       remotecmd = sprintf('touch %s; if [ -e %s ]; then rm -f %s && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
%       cmd1 = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host,'bg',false);
%       cmd = sprintf('timeout 20 %s',cmd1);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = system(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); drawnow;
%         return;
%       end
%       issuccess = contains(result,'SUCCESS');
%       isfailure = contains(result,'FAILURE');
%       if issuccess && ~isfailure,
%         hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       elseif ~issuccess && isfailure,
%         hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
%         return;
%       else
%         hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
%         return;
%       end
%       
%       % test that we can run bjobs
%       hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
%       remotecmd = 'bjobs';
%       cmd = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = system(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
%         return;
%       end
      hedit.String{end+1} = 'SUCCESS!'; 
      hedit.String{end+1} = ''; 
      hedit.String{end+1} = 'All tests passed. AWS Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end


    
  end
  
end
    