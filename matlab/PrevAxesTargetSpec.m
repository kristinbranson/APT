classdef PrevAxesTargetSpec
  % Value class holding both the identity and rendering data for the frozen/prev-axes frame.
  % Identity fields: iMov, frm, iTgt, gtmode.
  % Rendering fields: im, isrotated, xdata, ydata, A, tform, xlim, ylim, dxlim, dylim, prevAxesProps.
  %
  % Every PrevAxesTargetSpec in existence is fully valid: the constructor
  % takes name-value pairs for all properties and asserts isValid() before
  % returning.  The "unset" state is represented by [] rather than an
  % invalid object.

  properties (SetAccess=immutable)
    % Persisted fields
    iMov  % scalar positive integer movie index
    frm  % scalar positive integer frame number
    iTgt  % scalar positive integer target index
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
    prevAxesProps  % scalar struct holding desired XDir/YDir/XLim/YLim for the sidekick axes
  end  % properties

  methods
    function obj = PrevAxesTargetSpec(varargin)
      % Construct from name-value pairs.  Asserts validity before returning.
      props = struct(varargin{:});
      for fieldName = fieldnames(props)'
        obj.(fieldName{1}) = props.(fieldName{1});
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
                ~isempty(obj.ylim) && ...
                ~isempty(obj.prevAxesProps) ;
      assert(isValid, ...
             'PrevAxesTargetSpec: constructed object is not valid.  All required fields must be set.') ;
    end  % function

    function s = toStructForPersistence(obj)
      % Serialize identity fields + dxlim/dylim to a struct for saving to disk.
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode, ...
                 'dxlim', obj.dxlim, 'dylim', obj.dylim);
    end  % function

    function s = toStruct(obj)
      % Convert all properties to a scalar struct.
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode, ...
                 'dxlim', obj.dxlim, 'dylim', obj.dylim, ...
                 'im', obj.im, 'isrotated', obj.isrotated, ...
                 'xdata', obj.xdata, 'ydata', obj.ydata, ...
                 'A', obj.A, 'tform', obj.tform, ...
                 'xlim', obj.xlim, 'ylim', obj.ylim, ...
                 'prevAxesProps', obj.prevAxesProps) ;
    end  % function

    function result = char(obj)
      % Return a char array representation of this object.
      result = sprintf('PrevAxesTargetSpec(iMov=%d, frm=%d, iTgt=%d, gtmode=%d)', ...
                       obj.iMov, obj.frm, obj.iTgt, obj.gtmode);
    end  % function
  end  % methods

  methods (Static)
    function parsedInfo = parsePersistedStruct(s)
      % Parse a persisted struct into a plain struct with identity + offset fields.
      % Returns a struct with fields: iMov, frm, iTgt, gtmode, dxlim, dylim.
      % Returns [] for empty input.
      % Cannot return a PrevAxesTargetSpec since rendering fields are absent.
      if isempty(s)
        parsedInfo = [] ;
        return
      end
      parsedInfo = struct() ;
      if isfield(s, 'iMov')
        parsedInfo.iMov = s.iMov;
      else
        parsedInfo = [] ;
        return
      end
      if isfield(s, 'frm')
        parsedInfo.frm = s.frm;
      else
        parsedInfo = [] ;
        return
      end
      if isfield(s, 'iTgt')
        parsedInfo.iTgt = s.iTgt;
      else
        parsedInfo.iTgt = 1;
      end
      if isfield(s, 'gtmode')
        parsedInfo.gtmode = s.gtmode;
      else
        parsedInfo.gtmode = false;
      end
      if isfield(s, 'dxlim')
        parsedInfo.dxlim = s.dxlim;
      else
        parsedInfo.dxlim = [0 0];
      end
      if isfield(s, 'dylim')
        parsedInfo.dylim = s.dylim;
      else
        parsedInfo.dylim = [0 0];
      end
    end  % function

    function result = setprop(obj, varargin)
      % Make a new, independent, object of the class by replacing some field names
      % Usage: newObj = setprop(obj, prop1, val1, prop2, val2, ...)
      assert(mod(numel(varargin), 2) == 0, 'setprop:badArgs', 'Arguments must be property-value pairs');
      oldPairs = obj.toStruct() ;
      newPairs = struct(varargin{:});
      mergedPairs = oldPairs ;
      for fieldName = fieldnames(newPairs)'                                                                                                                              │
        mergedPairs.(fieldName{1}) = newPairs.(fieldName{1});                                                                                                                    │
      end       
      mergedPairsAsList = struct2pvs(mergedPairs) ;      
      result = PrevAxesTargetSpec(mergedPairsAsList{:}) ;
    end  % function

  end  % methods
end  % classdef
