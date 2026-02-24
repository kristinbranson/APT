classdef PrevAxesTargetSpec
  % Value class holding both the identity and rendering data for the frozen/prev-axes frame.
  % Identity fields: iMov, frm, iTgt, gtmode.
  % Rendering fields: im, isrotated, xdata, ydata, A, tform, xlim, ylim, dxlim, dylim.
  %
  % Every PrevAxesTargetSpec in existence is fully valid: the constructor
  % takes either a single struct or name-value pairs for all properties, and
  % asserts isValid() before returning.  The "unset" state is represented by
  % [] rather than an invalid object.

  properties (SetAccess=immutable)
    % Persisted fields
    iMov  % scalar positive integer movie index
    frm  % scalar positive integer frame number
    iTgt = 1  % scalar positive integer target index
    gtmode = false  % scalar logical, whether in GT mode
    dxlim = [0 0]  % adjustment to xlim as a result of user panning/zooming the sidekick axes, a 2-el row vec
    dylim = [0 0]  % adjustment to ylim as a result of user panning/zooming the sidekick axes, a 2-el row vec
  end

  properties (SetAccess=immutable, Transient)
    % Not persisted fields
    im  % the frame data, an image
    isrotated = false  % whether the target is shown rotated or not
    xdata  % a two-element row vector like XData for an image() gobject
    ydata  % a two-element row vector like YData for an image() gobject
    A  % a transformation matrix used to transform data in im into what is shown on-screen, or []
    tform  % a transformation object related to A, or []
    xlim  % the computed x-axis limits to use when showing the target, a 2-el row vec
    ylim  % the computed y-axis limits to use when showing the target, a 2-el row vec
  end  % properties

  methods
    function obj = PrevAxesTargetSpec(varargin)
      % Construct from a single struct or from name-value pairs.  Asserts validity before returning.
      if nargin == 1 && isstruct(varargin{1})
        props = varargin{1} ;
      else
        props = struct(varargin{:}) ;
      end
      myProps = properties(obj) ;
      for fieldName = fieldnames(props)'
        if ismember(fieldName{1}, myProps)
          obj.(fieldName{1}) = props.(fieldName{1}) ;
        end
      end
      isValid = isScalarFiniteNonneg(obj.iMov) && ...
                isScalarFiniteNonneg(obj.frm) && ...
                isScalarFiniteNonneg(obj.iTgt) && ...
                isscalar(obj.gtmode) && islogical(obj.gtmode) && ...
                isscalar(obj.isrotated) && islogical(obj.isrotated) && ...
                ~isempty(obj.dxlim) && ...
                ~isempty(obj.dylim) && ...
                ~isempty(obj.im) && ...
                ~isempty(obj.xdata) && ...
                ~isempty(obj.ydata) && ...
                ~isempty(obj.xlim) && ...
                ~isempty(obj.ylim) ;
      assert(isValid, ...
             'PrevAxesTargetSpec: constructed object is not valid.  All required fields must be set.') ;
    end  % function

    function s = struct(obj)
      % Convert all properties to a scalar struct.
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode, ...
                 'dxlim', obj.dxlim, 'dylim', obj.dylim, ...
                 'im', obj.im, 'isrotated', obj.isrotated, ...
                 'xdata', obj.xdata, 'ydata', obj.ydata, ...
                 'A', obj.A, 'tform', obj.tform, ...
                 'xlim', obj.xlim, 'ylim', obj.ylim) ;
    end  % function
  end  % methods

  methods (Static)
    function result = setprop(obj, varargin)
      % Make a new, independent, object of the class by replacing some field names
      % Usage: newObj = setprop(obj, prop1, val1, prop2, val2, ...)
      assert(mod(numel(varargin), 2) == 0, 'setprop:badArgs', 'Arguments must be property-value pairs') ;
      oldPairs = struct(obj) ;
      newPairs = struct(varargin{:}) ;
      mergedPairs = oldPairs ;
      for fieldName = fieldnames(newPairs)'
        mergedPairs.(fieldName{1}) = newPairs.(fieldName{1});
      end
      result = PrevAxesTargetSpec(mergedPairs) ;
    end  % function

  end  % methods
end  % classdef
