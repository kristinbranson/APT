function result = isenum_from_class_name(class_name) 
    mc = meta.class.fromName(class_name) ;
    result = logical(mc.Enumeration) ;
end
