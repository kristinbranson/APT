classdef PropertiesGUIProp < handle
  properties
    Field % char, fieldname on struct/obj
    DispName % displayname in table
    Type % {cellstr 'unsigned' 'signed' 'string' 'color' 'font' 'date' 'float' 'boolean' 'folder 'file' 'password' 'IPAddress'}
    isEditable % scalar logical
    Description % freeform text
    ParamViz % optional, char concrete classname for ParameterVisualization subclass
    DefaultValue 
    Value    
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
        dfltval,val,prmViz)
      obj.Field = fld;
      obj.DispName = dispname;
      obj.Type = type;
      obj.isEditable = editable;
      obj.Description = desc;
      obj.DefaultValue = dfltval;      
      obj.Value = val;
      obj.ParamViz = prmViz;
    end
  end
end
    
    