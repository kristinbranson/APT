classdef CorePrevAxesTargetSpec
  % Value class holding just the parts of PrevAxesTargetSpec that are persisted:
  % iMov, frm, iTgt, gtmode, dxlim, and dylim.
  %
  % Every CorePrevAxesTargetSpec in existence is fully valid: the constructor
  % takes a PrevAxesTargetSpec, a single struct, or name-value pairs, and
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
    azimuth = 0  % in-plane rotation angle (degrees) of the prev axes in frozen mode
  end

  methods
    function obj = CorePrevAxesTargetSpec(varargin)
      % Construct from a PrevAxesTargetSpec, a single struct, or name-value pairs.
      % Asserts validity before returning.
      if nargin == 1 && isa(varargin{1}, 'PrevAxesTargetSpec')
        source = varargin{1} ;
        obj.iMov = source.iMov ;
        obj.frm = source.frm ;
        obj.iTgt = source.iTgt ;
        obj.gtmode = source.gtmode ;
        obj.dxlim = source.dxlim ;
        obj.dylim = source.dylim ;
        obj.azimuth = source.azimuth ;
      else
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
      end
      isValid = isScalarFiniteNonneg(obj.iMov) && ...
                isScalarFiniteNonneg(obj.frm) && ...
                isScalarFiniteNonneg(obj.iTgt) && ...
                isscalar(obj.gtmode) && islogical(obj.gtmode) && ...
                ~isempty(obj.dxlim) && ...
                ~isempty(obj.dylim) ;
      assert(isValid, ...
             'CorePrevAxesTargetSpec: constructed object is not valid.  All required fields must be set.') ;
    end  % function

    function s = struct(obj)
      % Serialize identity fields + dxlim/dylim to a struct for saving to disk.
      s = struct('iMov', obj.iMov, 'frm', obj.frm, 'iTgt', obj.iTgt, 'gtmode', obj.gtmode, ...
                 'dxlim', obj.dxlim, 'dylim', obj.dylim, 'azimuth', obj.azimuth);
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
      result = CorePrevAxesTargetSpec(mergedPairs) ;
    end  % function

  end  % methods
end  % classdef
