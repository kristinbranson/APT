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
    function obj = PropertiesGUIProp(fld,dispname,type,editable,desc,...
        dfltval,val,prmViz,level,rqts,visible,affectsTraining)
      obj.Field = fld;
      obj.DispName = dispname;
      obj.Type = type;
      obj.isEditable = editable;
      obj.Description = desc;
      obj.DefaultValue = dfltval;      
      obj.Value = val;
      obj.ParamViz = prmViz;
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
end
    
    