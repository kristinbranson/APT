classdef VirtualLine < matlab.mixin.Copyable
  properties
    XData = nan
    YData = nan
    Color = [0 0 0]
    Marker = 'o'
    MarkerSize = 6
    LineWidth = 0.5
    UserData = []
    HitTest = 'on'
    PickableParts = 'all'
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
