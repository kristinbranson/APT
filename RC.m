classdef RC 
% Configuration file

  properties (Constant)
    FILE = fullfile(APT.Root,'.apt.mat');
  end
  
  methods (Static)
    
    % AL 20150606: Currently every getprop involves a filesystem read
    function v = getprop(name)
      rc = RC.load;
      if isfield(rc,name)
        v = rc.(name);
      else
        v = [];
      end
    end
    
    % AL 20150606: Currently every setprop involves a filesystem write
    function saveprop(name,v)
      tmp = struct(name,v); %#ok<NASGU>
      if exist(RC.FILE,'file')==2
        save(RC.FILE,'-append','-struct','tmp');
      else
        save(RC.FILE,'-struct','tmp');
      end
    end      
    
  end
  
  methods (Static,Hidden)
    
    function rc = load
      if exist(RC.FILE,'file')==2      
        rc = load(RC.FILE);
      else
        rc = struct();
      end
    end
        
  end
  
end