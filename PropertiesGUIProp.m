classdef PropertiesGUIProp 
  properties
    Field % char, fieldname on struct/obj
    DispName % displayname in table
    Type % {cellstr 'unsigned' 'signed' 'string' 'color' 'font' 'date' 'float' 'boolean' 'folder 'file' 'password' 'IPAddress'}
    isEditable % scalar logical
    Description % freeform text
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
  end
  methods 
    function obj = PropertiesGUIProp(fld,dispname,type,editable,desc,dfltval,val)
      obj.Field = fld;
      obj.DispName = dispname;
      obj.Type = type;
      obj.isEditable = editable;
      obj.Description = desc;
      obj.DefaultValue = dfltval;
      obj.Value = val;
    end
  end
end
    
    