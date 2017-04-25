classdef PropertiesGUIProp 
  properties
    Field % char, fieldname on struct/obj
    DispName % displayname in table
    Type % 'unsigned', 'signed', etc
    isEditable % scalar logical
    Description % freeform text
  end
  methods 
    function obj = PropertiesGUIProp(fld,dispname,type,editable,desc)
      obj.Field = fld;
      obj.DispName = dispname;
      obj.Type = type;
      obj.isEditable = editable;
      obj.Description = desc;
    end
  end
end
    
    