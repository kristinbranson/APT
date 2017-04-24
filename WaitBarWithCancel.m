classdef WaitBarWithCancel < handle
  
  properties
    hWB % waitbar handle
    isCancel % scalar logical
    
    msgPat % if nonempty, apply this printf-style pat for msgs
  end
  
  methods
    
    function obj = WaitBarWithCancel(title,varargin)
      h = waitbar(0,'','Name',title,varargin{:},'Visible','off',...
        'CreateCancelBtn',@(s,e)obj.cbkCancel(s,e));
      hTxt = findall(h,'type','text');
      hTxt.Interpreter = 'none';
      
      obj.hWB = h;
      obj.isCancel = false;
    end
    
    function delete(obj)
      delete(obj.hWB);
    end
    
    function cbkCancel(obj,s,e) %#ok<INUSD>
      obj.isCancel = true;
    end
    
  end
  
  methods

    function startCancelablePeriod(obj,msg,varargin)
      % Start a new cancelable period with waitbar message msg.
      
      obj.isCancel = false;
      
      if ~isempty(obj.msgPat)
        msg = sprintf(obj.msgPat,msg);
      end
      obj.hWB.Visible = 'on';      
      waitbar(0,obj.hWB,msg,varargin{:});
    end
      
    function tfCancel = updateFrac(obj,frac)
      tfCancel = obj.isCancel;
      waitbar(frac,obj.hWB);
    end
    
    function endCancelablePeriod(obj)
      obj.hWB.Visible = 'off';      
    end    
      
  end
    
  
end