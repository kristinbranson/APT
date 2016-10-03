classdef APT 
  
  properties (Constant)
    Root = fileparts(mfilename('fullpath'));    
    MANIFESTFILE = 'Manifest.txt';    
    Manifest = lclReadManifest( fullfile(APT.Root,APT.MANIFESTFILE) );    
    SnapshotScript = fullfile(APT.Root,'repo_snapshot.sh');   
    
    BUILDSNAPSHOTFILE = 'build.snapshot';
    BUILDSNAPSHOTFULLFILE = fullfile(APT.Root,APT.BUILDSNAPSHOTFILE);
    
    BUILDMCCFILE = 'build.mcc';
    BUILDMCCFULLFILE = fullfile(APT.Root,APT.BUILDMCCFILE);    
  end
  
  methods (Static)
  
    function [p,jp] = getpath()
      % p: cellstr, path entries      
      % jp: cellstr, javapath entries
      
      m = APT.Manifest;      
      jctroot = m.jctrax;
      if isfield(m,'cameracalib')
        camroot = m.cameracalib;
      else
        camroot = '';
      end
      root = APT.Root;
      campath = genpath(camroot);
      campath = regexp(campath,pathsep,'split');
      campath = campath(~cellfun(@isempty,campath));
      p = { ...
        root; ...
        fullfile(root,'misc'); ...
        fullfile(root,'private_imuitools'); ...
        fullfile(root,'user'); ...
        fullfile(root,'YAMLMatlab_0.4.3'); ...
        fullfile(root,'JavaTableWrapper'); ...
        fullfile(root,'propertiesGUI'); ...
        fullfile(root,'treeTable'); ...
        fullfile(jctroot,'filehandling'); ...
        fullfile(jctroot,'misc'); ...
        };
      p = [p;campath(:)];
      
      jp = {...
        fullfile(APT.Root,'JavaTableWrapper','+uiextras','+jTable','UIExtrasTable.jar'); ...
        fullfile(APT.Root,'YAMLMatlab_0.4.3','external','snakeyaml-1.9.jar')};
    end
    
    function setpath()
      [p,jp] = APT.getpath();
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
      javaaddpath(jp);
    end
    
    function s = codesnapshot
      % This method assumes that the user has set their path using
      % APT.setpath (so that the Manifest correclty reflects
      % dependencies). Do a quick+dirty check of this assumption.
      grf = which('get_readframe_fcn');
      manifest = APT.Manifest;
      if ~isequal(fileparts(grf),fullfile(manifest.jctrax,'filehandling'))
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
    
    function build()
      % build()
      
      proj = 'APTCluster';
            
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
      buildIfo.singlethreaded = {'-R' '-singleCompThread'};      

      pth = APT.getpath();
      pth = pth(:);
      Ipth = [repmat({'-I'},numel(pth),1) pth];
      Ipth = Ipth';      
      aptroot = APT.Root;
      cprroot = CPR.Root;
      
      outdir = fullfile(aptroot,proj);
      if exist(outdir,'dir')==0
        fprintf('Creating output dir: %s\n',outdir);
        [outdirparent,outdirbase] = fileparts(outdir);
        [tf,msg] = mkdir(outdirparent,outdirbase);
        if ~tf
          error('APT:dir','Could not make output dir: %s',msg);
        end
      end
      mccargbase = {...
        '-W' 'main',...
        '-w','enable',...
        '-T','link:exe',...
        '-d',fullfile(aptroot,proj),...
        '-v',...
        fullfile(aptroot,[proj '.m']),...
        Ipth{:},...
        '-a',fullfile(aptroot,'config.default.yaml'),...
        '-a',fullfile(cprroot,'misc','CPRLabelTracker.m'),...
        '-a',fullfile(aptroot,'LabelerGUI_lnx.fig'),...
        '-a',fullfile(cprroot,'param.example.yaml'),...
        '-a',fullfile(aptroot,'YAMLMatlab_0.4.3','external','snakeyaml-1.9.jar'),...
        '-a',fullfile(aptroot,'JavaTableWrapper','+uiextras','+jTable','UIExtrasTable.jar')};
        
      bldnames = fieldnames(buildIfo);
      for bld=bldnames(:)',bld=bld{1}; %#ok<FXSET>
        fprintf('Building: %s...\n',bld);
        pause(2);
        
        extraMccArgs = buildIfo.(bld);
        projfull = [proj '_' bld];
        extraMccArgs(end+1:end+2) = {'-o' projfull};
        mccargs = [mccargbase(:)' extraMccArgs(:)'];
        
        fprintf('Writing mcc args to file: %s...\n',APT.BUILDMCCFULLFILE);
        cellstrexport(mccargs,APT.BUILDMCCFULLFILE);
        
        today = datestr(now,'yyyymmdd');
        fprintf('BEGIN BUILD on %s\n',today);
        pause(2.0);
        mcc(mccargs{:});
        
        % postbuild
        mnfst = APT.Manifest;
        bindir = fullfile(mnfst.build,today);
        if exist(bindir,'dir')==0
          fprintf('Creating bin dir %s...\n',bindir);
          [succ,msg] = mkdir(bindir);
          if ~succ
            error('APT:build','Failed to create bin dir: %s\n',msg);
          end
        end
        fprintf('Moving binaries + build artifacts into: %s\n',bindir);
        % move buildmcc file, buildsnapshot file into bindir with name change
        % move binaries
        binsrc = fullfile(aptroot,proj,projfull);
        bindst = fullfile(bindir,proj,projfull);
        runsrc = fullfile(aptroot,proj,['run_' projfull '.sh']);
        rundst = fullfile(bindir,proj,['run_' projfull '.sh']);
        mccsrc = APT.BUILDMCCFULLFILE;
        mccdst = fullfile(bindir,proj,[projfull '.' APT.BUILDMCCFILE]);
        
        if exist(fullfile(bindir,proj),'dir')==0
          fprintf('Creating build dir %s...\n',fullfile(bindir,proj));
          [succ,msg] = mkdir(bindir,proj);
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
      
      sssrc = APT.BUILDSNAPSHOTFULLFILE;
      ssdst = fullfile(bindir,proj,APT.BUILDSNAPSHOTFILE);
      APT.buildmv(sssrc,ssdst);
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

function s = lclReadManifest(fname)
if exist(fname,'file')==0
  error('APT:Manifest','Cannot find Manifest file ''%s''. Please copy from Manifest.txt.sample and edit for your machine.',fname);
end
tmp = importdata(fname);
tmp = regexp(tmp,',','split');
tmp = cat(1,tmp{:});
s = cell2struct(tmp(:,2),tmp(:,1));
end