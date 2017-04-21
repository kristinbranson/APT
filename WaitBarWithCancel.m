classdef WaitBarWithCancel < handle
  
  properties
    hWB % waitbar handle
    isCancel % scalar logical
  end
  
  methods
    
    function obj = WaitBarWithCancel(msg,varargin)
      h = waitbar(0,msg,varargin{:},'CreateCancelBtn',@(s,e)obj.cbkCancel(s,e));
      hTxt = findall(h,'type','text');
      hTxt.Interpreter = 'none';
      
      obj.hWB = h;
      obj.isCancel = false;
    end
    
    function delete(obj)
      delete(obj.hWB);
    end
    
  end
  
  methods
    
    function cbkCancel(obj,s,e) %#ok<INUSD>
      obj.isCancel = true;
    end
    
    function tfCancel = update(obj,frac,varargin)
      % tfCancel = updateRaw(obj,frac)
      % tfCancel = updateRaw(obj,frac,msg)
      tfCancel = obj.isCancel;
      waitbar(frac,obj.hWB,varargin{:});
    end
        
  end
    
  
end