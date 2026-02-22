classdef VirtualText < matlab.mixin.Copyable
  properties
    Position = [nan nan 1]
    String = ''
    Color = [0 0 0]
    FontSize = 10
    FontName = 'Helvetica'
    FontWeight = 'normal'
    FontAngle = 'normal'
    Visible = 'on'
    PickableParts = 'none'
    Tag = ''
  end
  methods
    function set(obj, varargin)
      if nargin == 2 && isstruct(varargin{1})
        s = varargin{1};
        flds = fieldnames(s);
        for k = 1:numel(obj)
          for j = 1:numel(flds)
            obj(k).(flds{j}) = s.(flds{j});
          end
        end
      else
        for k = 1:numel(obj)
          for j = 1:2:numel(varargin)
            obj(k).(varargin{j}) = varargin{j+1};
          end
        end
      end
    end  % function

    function val = get(obj, propName)
      val = obj.(propName);
    end  % function
  end  % methods
end % classdef
