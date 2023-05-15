function result = does_class_have_static_method_named(method_name, class_name)
    mc = meta.class.fromName(class_name) ;
    method_from_index = mc.MethodList ;
    is_static_from_index = arrayfun(@(method)(logical(method.Static)), method_from_index) ;
    method_from_static_index = method_from_index(is_static_from_index) ;
    method_name_from_static_index = {method_from_static_index.Name} ;
    result = any(strcmp(method_name, method_name_from_static_index)) ;
end
