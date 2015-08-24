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
        };
    end
    
    function setpath()
      p = APT.getpath();
      addpath(p{:},'-begin');
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