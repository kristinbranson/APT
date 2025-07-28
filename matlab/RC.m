classdef RC 
% Configuration file

  properties (Constant)
    FILE = lclInitFile();
  end
  
  methods (Static)
    
    function v = getpropdefault(name,dfltval)
      % get prop if it exists; if not, set prop to default val and return
      % it
      
      rc = RC.load;
      if isfield(rc,name)
        v = rc.(name);
      else
        RC.saveprop(name,dfltval);
        v = dfltval;
      end
    end
    
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
      file = RC.FILE;
      if isempty(file)
        % none
      else
        tmp = struct(name,v); %#ok<NASGU>
        if exist(file,'file')==2
          save(file,'-append','-struct','tmp');
        else
          save(file,'-struct','tmp');
        end
      end
    end  
    
    function reset()
      % Clear entire RC contents
      file = RC.FILE;
      if exist(file,'file')>0
        delete(file);
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

  properties
    rcmatfile = [];
  end

  methods
    function obj = RC()
      if ~exist(RC.FILE,'file'),
        tmp = struct;
        save(RC.FILE,'-struct','tmp');
        assert(exist(RC.FILE,'file'));
      end
      obj.rcmatfile = matfile(RC.FILE,'Writable',true);
    end
    function v = get(obj,name)
      v = [];
      try
        v = obj.rcmatfile.(name);
      end
    end
    function set(obj,name,v)
      obj.rcmatfile.(name) = v;
    end

  end
  
end

function f = lclInitFile()
if isdeployed
  f = [];
else
  f = fullfile(APT.Root,'.apt.mat'); 
end
end