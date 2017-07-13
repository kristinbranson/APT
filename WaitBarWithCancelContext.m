classdef WaitBarWithCancelContext < handle
  
  properties
    message = '' % "Base" message
    nobar = false % If true, no graphical fraction bar shown
    shownumden = false % If true, show fractional degree of completion eg (3/500) in message
    numerator = nan % Numerator if shownumden==true
    denominator = nan % Denominator (total number of units to run) if shownumden==true
  end  
  
  methods
    
    function obj = WaitBarWithCancelContext(msg,varargin)
      obj.message = msg;
      nvarg = numel(varargin);
      assert(mod(nvarg,2)==0);
      for iprop=1:2:nvarg
        prop = varargin{iprop};
        val = varargin{iprop+1};
        obj.(prop) = val;
      end
    end
    
    function m = fullmessage(objs)
      m = '';
      for i=1:numel(objs)
        if objs(i).shownumden
          mnew = sprintf('%s (%d/%d)',objs(i).message,objs(i).numerator,...
            objs(i).denominator);
        else
          mnew = objs(i).message;
        end
        if i==1
          m = mnew;
        else
          m = [m ': ' mnew]; %#ok<AGROW>
        end
      end
    end
    
  end
end