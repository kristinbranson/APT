classdef PrevAxesTargetSpec
  % Value class holding both the identity and rendering data for the frozen/prev-axes frame.
  % Identity fields: iMov, frm, iTgt, gtmode.
  % Rendering fields: im, isrotated, xdata, ydata, A, tform, xlim, ylim, dxlim, dylim, prevAxesProps.

  properties
    % Identity fields
    iMov = []
    frm = []
    iTgt = []
    gtmode = false
    % Rendering fields
    im = []  % the frame data, an image; or []
    isrotated = false  % whether the target is shown rotated or not
    xdata = []  % a two-element row vector like XData for an image() gobject; or []
    ydata = []  % a two-element row vector like YData for an image() gobject; or []
    A = []  % A transformation matrix used to transform data in im into what is shown on-screen; or []
    tform = []  % Not sure.  Presumably related to A.  Or [].
    xlim = []  % The computed x-axis limits to use when showing the target.  A 2-el row vec, or [].
    ylim = []  % The computed y-axis limits to use when showing the target.  A 2-el row vec, or [].
    dxlim = []  % Adjustment to xlim as a result of user panning/zooming the sidekick axes.  A 2-el row vec, or [].
    dylim = []  % Adjustment to ylim as a result of user panning/zooming the sidekick axes.  A 2-el row vec, or [].
    prevAxesProps = []  % Scalar struct holding desired XDir/YDir/XLim/YLim for the sidekick axes.  Or [].
  end  % properties

  methods
    function obj = PrevAxesTargetSpec()
      % Zero-arg constructor; all properties use defaults.
    end  % function

    function result = isValid(obj)
      % Returns true iff both the identity and rendering fields are populated.
      result = obj.isTargetSet() && ...
               ~isempty(obj.im) && ...
               ~isempty(obj.xdata) && ...
               ~isempty(obj.ydata) && ...
               ~isempty(obj.xlim) && ...
               ~isempty(obj.ylim) && ...
               ~isempty(obj.dxlim) && ...
               ~isempty(obj.dylim) && ...
               ~isempty(obj.prevAxesProps);
    end  % function

    function result = isTargetSet(obj)
      % Returns true iff the identity fields (iMov, frm, iTgt, gtmode) are populated.
      result = isScalarFiniteNonneg(obj.iMov) && ...
               isScalarFiniteNonneg(obj.frm) && ...
               isScalarFiniteNonneg(obj.iTgt) && ...
               isscalar(obj.gtmode) && islogical(obj.gtmode);
    end  % function

    function s = toStructForPersistence(obj)
      % Serialize identity fields + dxlim/dylim to a struct for saving to disk.
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode, ...
                 'dxlim', obj.dxlim, 'dylim', obj.dylim);
    end  % function

    function result = char(obj)
      % Return a char array representation of this object.
      if obj.isTargetSet()
        result = sprintf('PrevAxesTargetSpec(iMov=%d, frm=%d, iTgt=%d, gtmode=%d)', ...
                         obj.iMov, obj.frm, obj.iTgt, obj.gtmode);
      else
        result = 'PrevAxesTargetSpec(<unset>)';
      end
    end  % function
  end  % methods

  methods (Static)
    function obj = fromPersistedStruct(s)
      % Reconstruct from a persisted struct.  Sets identity fields + dxlim/dylim.
      % Old .lbl files lacking dxlim/dylim get [0 0] defaults.
      obj = PrevAxesTargetSpec();
      if isempty(s)
        return
      end
      if isfield(s, 'iMov')
        obj.iMov = s.iMov;
      end
      if isfield(s, 'frm')
        obj.frm = s.frm;
      end
      if isfield(s, 'iTgt')
        obj.iTgt = s.iTgt;
      else
        obj.iTgt = 1;
      end
      if isfield(s, 'gtmode')
        obj.gtmode = s.gtmode;
      else
        obj.gtmode = false;
      end
      if isfield(s, 'dxlim')
        obj.dxlim = s.dxlim;
      else
        obj.dxlim = [0 0];
      end
      if isfield(s, 'dylim')
        obj.dylim = s.dylim;
      else
        obj.dylim = [0 0];
      end
    end  % function
  end  % methods
end  % classdef
