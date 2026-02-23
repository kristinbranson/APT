classdef PrevAxesTarget
  % Value class holding the identity of the frozen/prev-axes frame.
  % Fields: iMov, frm, iTgt, gtmode.

  properties
    iMov = []
    frm = []
    iTgt = []
    gtmode = false
  end  % properties

  methods
    function obj = PrevAxesTarget(varargin)
      if nargin == 0
        % zero-arg: empty/invalid target
      elseif nargin == 4
        obj.iMov = varargin{1};
        obj.frm = varargin{2};
        obj.iTgt = varargin{3};
        obj.gtmode = varargin{4};
      else
        error('PrevAxesTarget:badArgs', 'PrevAxesTarget requires 0 or 4 arguments');
      end
    end  % function

    function result = isValid(obj)
      result = isScalarFiniteNonneg(obj.iMov) && ...
               isScalarFiniteNonneg(obj.frm) && ...
               isScalarFiniteNonneg(obj.iTgt) && ...
               isscalar(obj.gtmode) && islogical(obj.gtmode);
    end  % function

    function s = toStruct(obj)
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode);
    end  % function

    function result = char(obj)
      if obj.isValid()
        result = sprintf('PrevAxesTarget(iMov=%d, frm=%d, iTgt=%d, gtmode=%d)', ...
                         obj.iMov, obj.frm, obj.iTgt, obj.gtmode);
      else
        result = 'PrevAxesTarget(<invalid>)';
      end
    end  % function
  end  % methods

  methods (Static)
    function obj = fromStruct(s)
      if isempty(s)
        obj = PrevAxesTarget();
        return
      end
      if isfield(s, 'iTgt')
        iTgt = s.iTgt;
      else
        iTgt = 1;
      end
      if isfield(s, 'gtmode')
        gtmode = s.gtmode;
      else
        gtmode = false;
      end
      obj = PrevAxesTarget(s.iMov, s.frm, iTgt, gtmode);
    end  % function
  end  % methods
end  % classdef
