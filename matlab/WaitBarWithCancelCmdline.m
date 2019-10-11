classdef WaitBarWithCancelCmdline < handle
% Dummy waitbar cmdline only
  
  properties
    isCancel = false;
    cancelData = [];
    % Stack of running contexts. contexts(end) is the top of the stack.
    contexts = WaitBarWithCancelContext.empty(0,1)
  end
  
  methods
    
    function obj = WaitBarWithCancelCmdline(title,varargin)
    end
    
    function delete(obj)
    end
        
  end
  
  methods

    function startPeriod(obj,msg,varargin)
      % Start a new cancelable period with waitbar message msg. Push new
      % context onto stack.
      %
      % varargin: see options for WaitBarWithCancelContext
      
      newContext = WaitBarWithCancelContext(msg,'numerator',0,varargin{:});
      obj.contexts(end+1,1) = newContext; % push
      obj.updateMessage();
    end
      
    function tfCancel = updateFrac(obj,frac)
      % Update waitbar fraction in top context.
      assert(~isempty(obj.contexts));
      assert(~logical(obj.contexts(end).shownumden));
      tfCancel = false;
      fprintf('... new frac: %.3f\n',frac);
    end
    
    function tfCancel = updateFracWithNumDen(obj,numerator)
      % Update waitbar fraction in top context.
      assert(~isempty(obj.contexts));
      ctxt = obj.contexts(end);
      assert(logical(ctxt.shownumden));
      ctxt.numerator = numerator;
      %frac = numerator/ctxt.denominator;
      
      tfCancel = false;
      obj.updateMessage();
    end
    
    function endPeriod(obj)
      % End current/topmost context.
      assert(~isempty(obj.contexts));
      obj.contexts = obj.contexts(1:end-1,:); % pop
      obj.updateMessage();
    end
    
    function msg = cancelMessage(obj,msgbase)
      msg = '';
    end
    
  end
  
  methods (Access=private)
    function updateMessage(obj)
      fprintf('%s\n',obj.contexts.fullmessage());
    end
  end
  
end