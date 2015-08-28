classdef APT 
  
  properties (Constant)    
    Root = fileparts(mfilename('fullpath'));    
    MANIFESTFILE = 'Manifest.txt';    
    Manifest = lclReadManifest( fullfile(APT.Root,APT.MANIFESTFILE) );    
    SnapshotScript = fullfile(APT.Root,'repo_snapshot.sh');    
  end
  
  methods (Static)
  
    function p = getpath()
      m = APT.Manifest;      
      jctroot = m.jctrax;
      root = APT.Root;
      p = { ...
        root; ...
        fullfile(root,'misc');
        fullfile(root,'private_imuitools');
        fullfile(root,'YAMLMatlab_0.4.3');
        fullfile(jctroot,'filehandling'); ...
        fullfile(jctroot,'misc'); ...
        };
    end
    
    function setpath()
      p = APT.getpath();
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
      
      mlver = ver('MATLAB');
      if isunix && strcmp(mlver.Release,'(R2015a)')
        randomyamlfile = fullfile(APT.Root,'YAMLMatlab_0.4.3','Tests','Data','test_import','file1.yaml');
        ReadYaml(randomyamlfile);
      end
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
        
        if ispc
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
    
  end
  
end

function s = lclReadManifest(fname)
tmp = importdata(fname);
tmp = regexp(tmp,',','split');
tmp = cat(1,tmp{:});
s = cell2struct(tmp(:,2),tmp(:,1));
end