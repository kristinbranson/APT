classdef APT
  
  properties (Constant)
    Root = fileparts(mfilename('fullpath'));
    MANIFESTFILE = 'Manifest.txt';
    SnapshotScript = fullfile(APT.Root,'matlab','repo_snapshot.sh');
    
%     BUILDSNAPSHOTFILE = 'build.snapshot';
%     BUILDSNAPSHOTFULLFILE = fullfile(APT.Root,APT.BUILDSNAPSHOTFILE);
    
%     BUILDMCCFILE = 'build.mcc';
%     BUILDMCCFULLFILE = fullfile(APT.Root,APT.BUILDMCCFILE);

    DOCKER_REMOTE_HOST = ''; % Conceptually this prob belongs in DLBackEndClass
    
    % for now, hard-coded to use default loc for git
    WINSCPCMD = 'C:\Program Files\Git\usr\bin\scp.exe';
    WINSSHCMD = 'C:\Program Files\Git\usr\bin\ssh.exe';

    % hardcoded name of AWS security group
    AWS_SECURITY_GROUP = 'apt_dl';
    AMI = 'ami-0168f57fb900185e1';  
  end
  
  methods (Static)
    
    function p = getRoot()
      if isdeployed
        p = ctfroot;
      else
        p = fileparts(mfilename('fullpath'));
      end
    end
    
    function m = readManifest()

      % KB 20190422 - Manifest no longer needed but can be used by
      % power-users
      
      fname = fullfile(APT.Root,APT.MANIFESTFILE);
      if exist(fname,'file')==0
        m = struct;
      else
        tmp = importdata(fname);
        tmp = regexp(tmp,',','split');
        tmp = cat(1,tmp{:});
        m = cell2struct(tmp(:,2),tmp(:,1));
      end
      
      % overwrite these fields to default locs if read in from Manifest
      root = APT.Root;
      m.jaaba = fullfile(root,'external','JAABA');
      m.piotr = fullfile(root,'external','PiotrDollarToolbox');
      m.cameracalib = fullfile(root,'external','CameraCalibrationToolbox');      
    end
  
    function [p,jp] = getpath()
      % p: cellstr, path entries      
      % jp: cellstr, javapath entries
      
      m = APT.readManifest;
      
      root = APT.Root;
      mlroot = fullfile(root,'matlab');
      cprroot = fullfile(mlroot,'trackers','cpr');
      if isfield(m,'jaaba')
        jaabaroot = m.jaaba;
      elseif isfield(m,'jctrax')
        jaabaroot = m.jctrax;
      else
        error('APT:noPath','Cannot find ''jaaba'' Manifest specification.');
      end
      if isfield(m,'piotr')
        pdolroot = m.piotr;
      end      
      if isfield(m,'cameracalib')
        camroot = m.cameracalib;
      end
      
      if isempty(pdolroot)
        %warnstr = 'No ''piotr'' Manifest entry found; CPR tracking will be unavailable. See Manifest.sample.txt.';
        %warningNoTrace('APT:cpr',warnstr);
        %warndlg(warnstr,'CPR/Tracking dependency missing','modal');        
      end
      
      if verLessThan('matlab','9.3')
        visionpath = 'vision_pre17b';
      else
        visionpath = 'vision_postinc17b';
      end
      aptpath = { ...
        root; ...
        mlroot; ...
        fullfile(mlroot,'util'); ...
        fullfile(mlroot,'misc'); ...
        fullfile(mlroot,'private_imuitools'); ...
        fullfile(root,'external','netlab'); ...
        fullfile(mlroot,'user'); ...
        fullfile(mlroot,'user','orthocam'); ...
        fullfile(mlroot,'user','orthocam',visionpath); ...
        fullfile(mlroot,'YAMLMatlab_0.4.3'); ...
        fullfile(mlroot,'JavaTableWrapper'); ...
        fullfile(mlroot,'propertiesGUI'); ...
        fullfile(mlroot,'treeTable'); ...
        fullfile(mlroot,'jsonlab-1.2','jsonlab'); ...
        fullfile(root,'test'); ...
        fullfile(mlroot,'compute_landmark_features'); ...
        fullfile(mlroot,'compute_landmark_transforms'); ...
        };
      
      cprpath = { ...
        cprroot; ...
        fullfile(cprroot,'misc'); ...
        fullfile(cprroot,'video_tracking'); ...
        fullfile(cprroot,'jan'); ...
        fullfile(cprroot,'romain'); ...
        };
      
      dtpath = { ...
        fullfile(mlroot,'trackers','dt'); ...
        };

      jaabapath = { ...
        fullfile(jaabaroot,'filehandling'); ...
        fullfile(jaabaroot,'misc'); ...
        };
      
      pdolpath = genpath(pdolroot);
      pdolpath = regexp(pdolpath,pathsep,'split');
      pdolpath = pdolpath(:);
      tfRm = cellfun(@(x) ~isempty(regexp(x,'__MACOSX','once')) || ...
                          ~isempty(regexp(x,'\.git','once')) || ...
                          ~isempty(regexp(x,'[\\/]doc','once')) || ...
                          ~isempty(regexp(x,'PiotrDollarToolbox[\\/]external','once')) || ...
                          isempty(x), pdolpath);
      pdolpath(tfRm,:) = [];

      campath = genpath(camroot);
      campath = regexp(campath,pathsep,'split');
      campath = campath(~cellfun(@isempty,campath));
     
      p = [aptpath(:);jaabapath(:);cprpath(:);dtpath(:);pdolpath(:);campath(:)];
      
      jp = {...
        fullfile(root,'java','APTJava.jar'); ...
        fullfile(mlroot,'JavaTableWrapper','+uiextras','+jTable','UIExtrasTable.jar'); ...
        fullfile(mlroot,'YAMLMatlab_0.4.3','external','snakeyaml-1.9.jar'); ...
        fullfile(mlroot,'treeTable')};
    end
    
    function jaabapath = getjaabapath()
      m = APT.readManifest;
      jaabaroot = m.jaaba;
      jaabapath = { ...
        fullfile(jaabaroot,'filehandling'); ...
        fullfile(jaabaroot,'misc'); ...
        };
    end
    
    function setpath()
      
      [p,jp] = APT.getpath();
      addpath(fullfile(APT.Root,'matlab')); % for javaaddpathstatic
      cellfun(@javaaddpathstatic,jp);
      addpath(p{:},'-begin');
      
      % AL 20150824, testing of sha 1f65 on R2015a+Linux is reproducably
      % SEGV-ing from a fresh MATLAB start when calling the Labeler()
      % constructor. (Specifically: >> APT.setpath; >> q = Labeler();)
      % Trace is unintelligible, mostly dispatcher/interpreter.
      %
      % Commenting out the call to ReadYaml() on Labeler.m:Line250 (as well
      % as the following line, which depends on it) resolves the SEGV; so
      % does calling ReadYaml on any random Yaml file before calling
      % Labeler(). 
      %
      % Based on this, hypothesize possible issue with java/the Yaml
      % library, when loaded inside a class, etc. As a hack, call ReadYaml
      % here on a random file, which will load the Yaml stuff before the
      % Labeler is instantiated.
      
%       mlver = ver('MATLAB');
%       if isunix && strcmp(mlver.Release,'(R2015a)')
%         randomyamlfile = fullfile(APT.Root,'YAMLMatlab_0.4.3','Tests','Data','test_import','file1.yaml');
%         ReadYaml(randomyamlfile);
%       end

%       javaaddpath(jp);
    end
    
    function setpathsmart()
      % Don't set MATLAB path if it appears it is already set
      % "smart" in quotes, of course
      
      if isdeployed
        return;
      end
        
      [p,jp] = APT.getpath();
      if APT.matlabPathNotConfigured
        fprintf('Configuring your MATLAB path ...\n');
        addpath(p{:},'-begin');
      end
      cellfun(@javaaddpathstatic,jp);
      %MK 20190506 Add stuff to systems path for aws cli
      if ismac
        setenv('PATH',['/usr/local/bin:' getenv('PATH')]);
      end
    end
  
    function tf = matlabPathNotConfigured()
      tf = exist('moveMenuItemAfter','file')==0 || ...
        exist('ReadYaml','file')==0;
    end
    
    function pposetf = getpathdl()
      r = APT.Root;
      pposetf = fullfile(r,'deepnet');
    end
    
    function cacheDir = getdlcacheroot()
      
      m = APT.readManifest;
      if isfield(m,'dltemproot')
        cacheDir = m.dltemproot;
      else
        if ispc
          userDir = winqueryreg('HKEY_CURRENT_USER',...
            ['Software\Microsoft\Windows\CurrentVersion\' ...
            'Explorer\Shell Folders'],'Personal');
        else
          userDir = char(java.lang.System.getProperty('user.home'));
        end
        cacheDir = fullfile(userDir,'.apt');
      end
    end
    
    function s = codesnapshot
      % This method assumes that the user has set their path using
      % APT.setpath (so that the Manifest correclty reflects
      % dependencies). Do a quick+dirty check of this assumption.
      grf = which('get_readframe_fcn');
      manifest = APT.readManifest;
      if ~isequal(fileparts(grf),fullfile(manifest.jaaba,'filehandling'))
        warning('APT:manifest',...
          'Runtime path appears to differ from that specified by Manifest. Code snapshot is likely to be incorrect.');
      end
      
      if isunix
        script = APT.SnapshotScript;
        cmd = sprintf('%s -nocolor -brief %s',script,APT.Root);
        [~,s] = system(cmd);
        s = regexp(s,sprintf('\n'),'split');
        modules = fieldnames(manifest);        
        modules = setdiff(modules,'build');
        for i = 1:numel(modules)        
          mod = modules{i};
          cmd = sprintf('%s -nocolor -brief %s',script,manifest.(mod));
          [~,stmp] = system(cmd);
          stmp = regexp(stmp,sprintf('\n'),'split');
          s = [s(:);{''};sprintf('### %s',upper(mod));stmp(:)];
        end
      elseif ispc
        cdir = pwd;
        manifest.apt = APT.Root;
        modules = fieldnames(manifest);
        cmd1 = 'git log -n 1 --all --pretty=format:"%h%x09%d%x20%s"';
        cmd2 = 'git status --porcelain .';
        s = struct();
        try
          for mod = modules(:)',mod=mod{1}; %#ok<FXSET>
            cd(manifest.(mod))
            [~,s1] = system(cmd1);
            [~,s2] = system(cmd2);
            s.(mod) = {s1;s2};
          end
        catch ME
          fprintf(2,'Err taking snapshot: %s',ME.getReport());
          s = [];
        end
        cd(cdir);
      end      
    end
    
    function buildAPTCluster(varargin) 
      % OBSOLETE probably broken due to paths/reorg 20191009
      [incsinglethreaded,bindirname] = myparse(varargin,...
        'incsinglethreaded',true,...
        'bindirname',[]... % custom binary output dir, eg '20180709.feature.deeptrack'. Still located underneath Manifest:build dir
        );
      today = datestr(now,'yyyymmdd');
      if isempty(bindirname)
        bindirname = today;
      end
      
      if ~isequal(pwd,APT.Root)
        error('Run APT.build in the APT root directory (%s), because mcc is finicky about includes/adds, the ctf archive, etcetera.\n',APT.Root);
      end
                        
      % take snapshot + save it to snapshot file
      codeSSfname = APT.BUILDSNAPSHOTFULLFILE;
      fprintf('Taking code snapshot and writing to file: %s...\n',codeSSfname);
      codeSS = APT.codesnapshot();
      codeSS = cellstr(codeSS);
      cellstrexport(codeSS,codeSSfname);
      fprintf('... done with snapshot.\n');
      
      % Generate mcc args
      buildIfo = struct();
      buildIfo.multithreaded = {};
      if incsinglethreaded
        buildIfo.singlethreaded = {'-R' '-singleCompThread'};
      end

      pth = APT.getpath();
      pth = pth(:);
      Ipth = [repmat({'-I'},numel(pth),1) pth];
      Ipth = Ipth';      
      aptroot = APT.Root;
      mlroot = fullfile(aptroot,'matlab');
      % 20191008: paths below are now out of date with repo re-org. 
      % buildAPTCluster obsolete
      cprroot = fullfile(aptroot,'trackers','cpr');
      dtroot = fullfile(aptroot,'trackers','dt');
      jaabapath = APT.getjaabapath();
      Ipthjaaba = [repmat({'-I'},numel(jaabapath),1) jaabapath];
      Ipthjaaba = Ipthjaaba';      

      BUILDOUTDIR = 'APTCluster';

      outdir = fullfile(aptroot,BUILDOUTDIR);
      if exist(outdir,'dir')==0
        fprintf('Creating output dir: %s\n',outdir);
        [outdirparent,outdirbase] = fileparts(outdir);
        [tf,msg] = mkdir(outdirparent,outdirbase);
        if ~tf
          error('APT:dir','Could not make output dir: %s',msg);
        end
      end
      
      % AL20181011, R2016b. Building on the cluster complains that no 
      % licenses are avail for certain products (eg toolbox/controls) even 
      % though those products are unnecessary and even if those products 
      % are explicitly removed from the MATLAB path before building. The 
      % dependency analysis must be getting confused and adding those paths 
      % back in etc etc. 
      %
      % Building on a branson-ws works better. It could be that the cluster
      % has more installed toolboxes/products, some of which don't have 
      % licenses etc. 
      %
      % AVOID ADDING the -N compilation option if possible, it does cause
      % breakage without the auto dependency analysis to configure the path
      % or add dependencies etc. Building on a branson-WS does not require
      % the -N flag currently.
      %
      % Update 20181126. Building on the cluster with 16b still fails, but
      % it works with 18b.
      
      mccProjargs = struct();
      mccProjargs.APTCluster = { ...
        '-W','main',...
        '-w','enable',...
        '-T','link:exe',...
        '-d',fullfile(aptroot,BUILDOUTDIR),... %        '-v',...
        fullfile(aptroot,'APTCluster.m'),... %'-N' see note above 20181011
        Ipth{:},...
        '-a',fullfile(aptroot,'gfx'),...
        '-a',fullfile(aptroot,'config.default.yaml'),...
        '-a',fullfile(aptroot,InfoTimeline.TLPROPFILESTR),...
        '-a',fullfile(aptroot,'params_preprocess.yaml'),...
        '-a',fullfile(aptroot,'params_track.yaml'),...
        '-a',fullfile(aptroot,'params_postprocess.yaml'),...
        '-a',fullfile(aptroot,'landmark_features.yaml'),...        
        '-a',fullfile(aptroot,'misc','darkjet.m'),...
        '-a',fullfile(aptroot,'misc','lightjet.m'),...
        '-a',fullfile(cprroot,'params_cpr.yaml'),... %        '-a',fullfile(cprroot,'param.example.yaml'),...
        '-a',fullfile(cprroot,'misc','CPRLabelTracker.m'),...
        '-a',fullfile(cprroot,'misc','CPRBlurPreProc.m'),...
        '-a',fullfile(dtroot,'params_deeptrack_dlc.yaml'),...
        '-a',fullfile(dtroot,'params_deeptrack_unet.yaml'),...
        '-a',fullfile(dtroot,'params_deeptrack.yaml'),...
        '-a',fullfile(dtroot,'params_deeptrack_openpose.yaml'),...
        '-a',fullfile(dtroot,'params_deeptrack_mdn.yaml'),...
        '-a',fullfile(dtroot,'params_deeptrack_leap.yaml'),...
        '-a',fullfile(dtroot,'DeepTracker.m'),...        
        '-a',fullfile(aptroot,'LabelerGUI_lnx.fig'),... 
        '-a',fullfile(aptroot,'YAMLMatlab_0.4.3','external','snakeyaml-1.9.jar'),...
        '-a',fullfile(aptroot,'JavaTableWrapper','+uiextras','+jTable','UIExtrasTable.jar'),...
        '-a',fullfile(aptroot,'java','APTJava.jar')...       
        }; %#ok<CCAT>
      mccProjargs.GetMovieNFrames = {...
        '-W','main',...
        '-w','enable',...
        '-T','link:exe',...
        '-d',fullfile(aptroot,BUILDOUTDIR),... %        '-v',...
        fullfile(aptroot,'misc','GetMovieNFrames.m'),...
        Ipthjaaba{:}};
        
      bldnames = fieldnames(buildIfo);
      projs = fieldnames(mccProjargs);
      projs = projs(end:-1:1); % build GetMovieNFrames first
      mnfst = APT.readManifest;
      bindir = fullfile(mnfst.build,bindirname);
      if exist(bindir,'dir')==0
        fprintf('Creating bin dir %s...\n',bindir);
        [succ,msg] = mkdir(bindir);
        if ~succ
          error('APT:build','Failed to create bin dir: %s\n',msg);
        end
      end
      for bld=bldnames(:)',bld=bld{1}; %#ok<FXSET>
        for prj=projs(:)',prj=prj{1};
          projfull = [prj '_' bld];
          fprintf('Building: %s...\n',projfull);
          pause(2);

          extraMccArgs = buildIfo.(bld);  
          extraMccArgs(end+1:end+2) = {'-o' projfull};
          mccArgs = mccProjargs.(prj);
          mccArgs = [mccArgs(:)' extraMccArgs(:)'];
          fprintf('Writing mcc args to file: %s...\n',APT.BUILDMCCFULLFILE);
          cellstrexport(mccArgs,APT.BUILDMCCFULLFILE);
        
          fprintf('BEGIN BUILD on %s\n',today);
          pause(2.0);
          mcc(mccArgs{:});

          % postbuild          
          fprintf('Moving binaries + build artifacts into: %s\n',bindir);
          % move buildmcc file, buildsnapshot file into bindir with name change
          % move binaries
          binsrc = fullfile(aptroot,BUILDOUTDIR,projfull);
          bindst = fullfile(bindir,BUILDOUTDIR,projfull);
          runsrc = fullfile(aptroot,BUILDOUTDIR,['run_' projfull '.sh']);
          rundst = fullfile(bindir,BUILDOUTDIR,['run_' projfull '.sh']);
          mccsrc = APT.BUILDMCCFULLFILE;
          mccdst = fullfile(bindir,BUILDOUTDIR,[projfull '.' APT.BUILDMCCFILE]);
        
          if exist(fullfile(bindir,BUILDOUTDIR),'dir')==0
            fprintf('Creating build dir %s...\n',fullfile(bindir,BUILDOUTDIR));
            [succ,msg] = mkdir(bindir,BUILDOUTDIR);
            if ~succ
              error('APT:build','Failed to create build dir: %s\n',msg);
            end
          end      
          APT.buildmv(binsrc,bindst);
          APT.buildmv(runsrc,rundst);
          APT.buildmv(mccsrc,mccdst);
          fileattrib(bindst,'+x');
          fileattrib(rundst,'+x');
%         
%         mccExc = fullfile(aptroot,'mccExcludedFiles.log');
%         readme = fullfile(aptroot,'readme.txt');
%         if exist(mccExc,'file')>0
%           delete(mccExc);
%         end
%         if exist(readme,'file')>0
%           delete(readme);
%         end
        end
      end
      
      sssrc = APT.BUILDSNAPSHOTFULLFILE;
      ssdst = fullfile(bindir,BUILDOUTDIR,APT.BUILDSNAPSHOTFILE);
      APT.buildmv(sssrc,ssdst);
      
      % drop a token for matlab version
      if isunix
        mlver = version('-release');
        cmd = sprintf('touch %s',fullfile(bindir,BUILDOUTDIR,mlver));
        system(cmd);
      end
    end
    
    function s = settingssnapshot(settingsdir)
      assert(exist(settingsdir,'dir')>0,'Cannot find dir ''%s''.',settingsdir);
      script = FlyBubbleBaR.SnapshotScript;
      cmd = sprintf('%s -nocolor -brief %s',script,settingsdir);
      [~,s] = system(cmd);
      s = regexp(s,sprintf('\n'),'split');
      s = s(:);
    end
    
    function buildmv(src,dst)
      if exist(dst,'file')>0
        warning('APT:build','Overwriting existing file:  %s',dst);
      end
      [succ,msg] = movefile(src,dst);
      if ~succ
        error('FlyBubbleBaR:build','Failed to move file ''%s'' -> ''%s'': %s\n',...
          src,dst,msg);
      end
    end
    
  end
  
end
