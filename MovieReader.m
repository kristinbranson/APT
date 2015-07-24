classdef MovieReader < handle
  % MovieReader
  % Like VideoReader; wraps get_readframe_fcn
  
  properties
    filename = '';
    
    readFrameFcn = [];
    nframes = nan;
    info = [];
    nr = nan;
    nc = nan;
    
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
      
      ifo = obj.info;
      if isfield(ifo,'nr') && isfield(ifo,'nc')
        obj.nr = ifo.nr;
        obj.nc = ifo.nc;
      else
        im = obj.readFrameFcn(1);
        [obj.nr,obj.nc] = size(im);
      end
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
      obj.nr = nan;
      obj.nc = nan;
      
      obj.fid = nan;
    end
      
    function delete(obj)
      obj.close();
    end
    
  end
  
end

