classdef PrevAxesTargetCache
  % Value class holding derived rendering data for the prev-axes frame.
  % Fields: im, isrotated, xdata, ydata, A, tform, xlim, ylim, dxlim,
  % dylim, prevAxesProps.

  properties
    im = []
    isrotated = false
    xdata = []
    ydata = []
    A = []
    tform = []
    xlim = []
    ylim = []
    dxlim = []
    dylim = []
    prevAxesProps = []
  end  % properties

  methods
    function obj = PrevAxesTargetCache()
      % Zero-arg constructor; all properties use defaults.
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
