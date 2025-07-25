classdef PropertiesGUIProp < matlab.mixin.SetGet & matlab.mixin.Copyable
  
  properties
    
    % *IMPORTANT*: properties must not be handles; PropertiesGUIProp
    % objects must be fully-copied by the shallow copy() method provided by
    % matlab.mixin.Copyable!!!
    
    Field % char, fieldname on struct/obj
    DispName % displayname in table
    Type % {cellstr 'unsigned' 'signed' 'string' 'color' 'font' 'date' 'float' 'boolean' 'folder 'file' 'password' 'IPAddress'}
    isEditable % scalar logical
    Description % freeform text
    ParamViz % optional, char concrete classname for ParameterVisualization subclass
    DefaultValue 
    Value    
    Level = 'Important'
    Requirements = {}
    Visible = true
    AffectsTraining = true
    Index = nan % leaf number
    FullPath = '' % path in the parameters tree to this parameter
    UserData = [];
    
  end
  properties (Dependent)
    DispNameUse
  end
  methods
    function v = get.DispNameUse(obj)
      v = obj.DispName;
      if isempty(v)
        v = obj.Field;
      end
    end
    function set.Value(obj,val)
      type = obj.Type; %#ok<MCSUP>
      if ischar(type) % type can be a cell for enums
        switch type
          case 'unsigned'
            % Doesn't look required by propertiesGUI
            % val = uint64(val);
          case 'signed'
            % Doesn't look required by propertiesGUI
            % val = int64(val);
          case 'boolean'
            % REQUIRED by propertiesGUI
            val = logical(val);
        end
      end
      obj.Value = val;
    end
  end
  methods 
    function obj = PropertiesGUIProp(fullPath,fld,dispname,type,editable,desc,...
        dfltval,val,prmViz,level,rqts,visible,affectsTraining)
      obj.Field = fld;
      obj.DispName = dispname;
      obj.Type = type;
      obj.isEditable = editable;
      obj.Description = desc;
      obj.DefaultValue = dfltval;      
      obj.Value = val;
      obj.ParamViz = prmViz;
      obj.FullPath = fullPath;
      if isempty(level),
        level = 'Important';
      end
      obj.Level = PropertyLevelsEnum(level);
      if ischar(rqts) && ~isempty(rqts)
        obj.Requirements = {rqts};
      elseif iscellstr(rqts)
        obj.Requirements = rqts;
      end
      if exist('visible','var'),
        obj.Visible = visible;
      end
      if exist('affectsTraining','var'),
        obj.AffectsTraining = affectsTraining;
      end        
    end
    
    function addRequirement(obj,req)
      obj.Requirements{end+1} = req;
    end
  end

  methods (Static)

    function [obj,fnsused] = initFromStruct(s,fld)
 
      fns = {'FullPath','Field','DispName','Type','isEditable',...
        'Description','DefaultValue','Value','ParamViz',...
        'Level','Requirements','Visible','AffectsTraining'};

      defaults = struct;
      defaults.DispName = '';
      defaults.Type = '';
      defaults.isEditable = false;
      defaults.Description = '';
      defaults.DefaultValue = [];
      defaults.Value = [];
      defaults.ParamViz = '';
      defaults.FullPath = '';
      defaults.Level = '';
      defaults.Requirements = {};
      defaults.AffectsTraining = [];
      defaults.Visible = [];

      args = cell(size(fns));

      for i = 1:numel(fns),
        fn = fns{i};
        if strcmp(fn,'Field'),
          args{i} = fld;
          continue;
        end
        if isfield(s,fn),
          args{i} = s.(fn);
        else
          args{i} = defaults.(fn);
        end
      end
      fnsused = intersect(fns,fieldnames(s));
      obj = PropertiesGUIProp(args{:});

    end

  end
end
    
    
