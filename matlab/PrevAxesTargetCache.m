classdef PrevAxesTargetCache
  % Value class holding derived rendering data for the prev-axes frame.
  % Fields: im, isrotated, xdata, ydata, A, tform, xlim, ylim, dxlim,
  % dylim, prevAxesProps.

  properties
    im = []  % the frame data, an image; or []
    isrotated = false  % whether the target is shown rotated or not
    xdata = []  % a two-element row vector like XData for an image() gobject; or []
    ydata = []  % a two-element row vector like YData for an image() gobject; or []
    A = []  % A tranformation matrix used to transform data in im into what is shown on-screen; or []
    tform = []  % Not sure.  Presumably related to A.  Or [].
    xlim = []  % The computed x-axis limits to use when showing the target.  A 2-el row vec, or [].
    ylim = []  % The computed y-axis limits to use when showing the target.  A 2-el row vec, or [].
    dxlim = []  % Adjustment to xlim as a result of user panning/zooming the sidekick axes.  A 2-el row vec, or [].
    dylim = []  % Adjustment to ylim as a result of user panning/zooming the sidekick axes.  A 2-el row vec, or [].
    prevAxesProps = []  % Scalar struct holding desired XDir/YDir/XLim/YLim for the sidekick axes.  Or [].
                        % (why do we need XLim/YLim here?  Isn't it redundant?)
  end  % properties

  methods
    function obj = PrevAxesTargetCache()
      % Zero-arg constructor; all properties use defaults.
    end  % function

    function result = isValid(obj)
      result = ~isempty(obj.im) && ...
               ~isempty(obj.xdata) && ...
               ~isempty(obj.ydata) && ...
               ~isempty(obj.xlim) && ...
               ~isempty(obj.ylim) && ...
               ~isempty(obj.dxlim) && ...
               ~isempty(obj.dylim) && ...
               ~isempty(obj.prevAxesProps);
    end  % function

    function result = char(obj)
      if isempty(obj.im)
        result = 'PrevAxesTargetCache(<empty>)';
      else
        result = sprintf('PrevAxesTargetCache(im=%dx%d, isrotated=%d)', ...
                         size(obj.im, 1), size(obj.im, 2), obj.isrotated);
      end
    end  % function
  end  % methods
end  % classdef
