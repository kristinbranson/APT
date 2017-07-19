classdef WaitBarWithCancel < handle
% Cancelable waitbar
%
% When using the builtin waitbar-with-CreateCancelBtn, the client must
% poll state set by the asynchronous cancel callback to determine if a
% cancel has occurred. WaitBarWithCancel incorporates this polling so it
% is automatically done at every update-how-much-work-remains.
%
% The bigger picture. Suppose you have a function compute() which does
% some potentially time-consuming work:
%
% results = compute(...args...)
%
% Suppose you want compute() to be optionally cancelable. You could pass
% an optional flag, but then you will typically need an addnl output arg
% to indicate whether a cancel occurred:
%
% [tfcanceled,results] = compute(...args...,'Cancelable',true);
% if tfcanceled
%   % do something
% else
%   % proceed with results
% end
%
% This has two drawbacks: i) the flag is extraneous in the case when
% 'Cancelable' is false; ii) if this is a nested computation, then each
% fcn in the calling stack may also need a tfcanceled flag to pass up
% the chain to alert calling functions that the cancel has occurred.
% These flags will not contain information on the nature of the cancel
% without a lot of work.
%
% Alternatively, we can pass in a WaitBarWithCancel obj:
%
% wbObj = WaitBarWithCancel(...);
% results = compute(...args...,'wbObj',wbObj);
% tfCancel = wbObj.isCancel;
% delete(wbObj);
% if tfCancel
%   % ... do something ...
% else
%   proceed with results
% end
%
% Advantages: i) Only callsites that want cancel-ability will have to
% deal with wbObj or .isCancel; ii) a single wbObj carrying can be passed
% up the call chain carrying any cancel metadata.
%
% Hmm, it's not a slam dunk but go with it for now.
% So, a function will be "cancel-enabled" if/when:
% - You can optionally pass in a WaitBarWithCancel (eg as 'wbObj')
% - If provided, the function uses the wbObj and early-returns gracefully
% if user Cancels. What exactly happens on Cancel is function-dependent
% and must be documented (eg a computation is partially carried out etc).
% If wbObj is not provided, the function always runs to completion.
  
  properties
    hWB % waitbar handle
    hTxt % handle to text obj in hWB
    hBar % handle to hgjavacomponent graphical waitbar
    isCancel % scalar logical
    cancelData % optional data set on main thread by clients for passing info upstream
    
    % Stack of running contexts. contexts(end) is the top of the stack.
    contexts = WaitBarWithCancelContext.empty(0,1)
  end
  
  methods
    
    function obj = WaitBarWithCancel(title,varargin)
      
      % cancelDisabled. Sometimes a function is Cancel-enabled but some of
      % its callsites/clients are not. In this case, cancellability can be
      % turned off so that a plain waitbar can still be used.
      iprop = find(strcmpi(varargin,'cancelDisabled'));
      if ~isempty(iprop)
        ival = iprop+1;
        cancelDisable = varargin{ival};
        varargin([iprop ival]) = [];
      else
        cancelDisable = false;
      end
      
      if cancelDisable 
        h = waitbar(0,'','Name',title,varargin{:},'Visible','off');
      else
        h = waitbar(0,'','Name',title,varargin{:},'Visible','off',...
          'CreateCancelBtn',@(s,e)obj.cbkCancel(s,e));
      end
      hText = findall(h,'type','text');
      hText.Interpreter = 'none';
      
      obj.hWB = h;
      obj.hTxt = hText;
      obj.hBar = findall(h,'type','hgjavacomponent');
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

    function startPeriod(obj,msg,varargin)
      % Start a new cancelable period with waitbar message msg. Push new
      % context onto stack.
      %
      % varargin: see options for WaitBarWithCancelContext
      
      %obj.isCancel = false;
      newContext = WaitBarWithCancelContext(msg,'numerator',0,varargin{:});
      obj.contexts(end+1,1) = newContext; % push
      waitbar(0,obj.hWB);
      obj.updateMessage();
      obj.updateShowBar();
      obj.hWB.Visible = 'on';
    end
      
    function tfCancel = updateFrac(obj,frac)
      % Update waitbar fraction in top context.
      assert(~isempty(obj.contexts));
      assert(~logical(obj.contexts(end).shownumden));
      tfCancel = obj.isCancel;
      waitbar(frac,obj.hWB); % msg is unchanged
    end
    
    function tfCancel = updateFracWithNumDen(obj,numerator)
      % Update waitbar fraction in top context.
      assert(~isempty(obj.contexts));
      ctxt = obj.contexts(end);
      assert(logical(ctxt.shownumden));
      ctxt.numerator = numerator;
      frac = numerator/ctxt.denominator;
      
      tfCancel = obj.isCancel;
      waitbar(frac,obj.hWB); 
      obj.updateMessage();
    end
    
    function endPeriod(obj)
      % End current/topmost context.
      assert(~isempty(obj.contexts));
      obj.contexts = obj.contexts(1:end-1,:); % pop
      obj.updateMessage();
      obj.updateShowBar();
      if isempty(obj.contexts)
        obj.hWB.Visible = 'off';
      end
    end
    
    function msg = cancelMessage(obj,msgbase)
      cdata = obj.cancelData;
      if isempty(cdata)
        msg = [msgbase '.'];
      else
        msg = sprintf('%s: %s',msgbase,cdata.msg);
      end
    end
    
  end
  
  methods (Access=private)
    function updateMessage(obj)
      obj.hTxt.String = obj.contexts.fullmessage();
    end
    function updateShowBar(obj)
      onoff = onIff(~isempty(obj.contexts) && ~obj.contexts(end).nobar);
      obj.hBar.Visible = onoff;
    end
  end
  
end