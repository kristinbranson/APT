classdef MovieReader < handle
  % MovieReader
  % Like VideoReader; wraps get_readframe_fcn
  
  properties
    filename = '';
    
    readFrameFcn = [];
    nframes = nan;
    info = [];
    
    fid = nan;
  end
  
  properties (Dependent)
    isOpen
  end
  
  methods
    function v = get.isOpen(obj)
      v = ~isnan(obj.fid);
    end
  end
  
  methods
    
    function obj = MovieReader
      % none
    end
        
    function open(obj,fname)
      assert(exist(fname,'file')>0,'Movie ''%s'' not found.',fname);
      
      if obj.isOpen
        obj.close();
      end
      
      obj.filename = fname;      
      [obj.readFrameFcn,obj.nframes,obj.fid,obj.info] = get_readframe_fcn(obj.filename);
    end
    
    function varargout = readframe(obj,i)
      assert(obj.isOpen(),'Movie is not open.');
      [varargout{1:nargout}] = obj.readFrameFcn(i);
    end
    
    function close(obj)
      if obj.fid>0
        fclose(obj.fid);
      end
      
      obj.readFrameFcn = [];
      obj.nframes = nan;
      obj.info = [];
      obj.fid = nan;
    end
      
    function delete(obj)
      obj.close();
    end
    
  end
  
end

