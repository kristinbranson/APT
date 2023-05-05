function property_names = property_names_satisfying_predicate(object, predicate_function)
    % Return a list of all the property names for the class that
    % satisfy the predicate, in the order they were defined in the
    % classdef.  predicate_function should be a function that
    % returns a logical when given a meta.Property object.
    
    mc = metaclass(object) ;
    all_class_properties = mc.Properties ;
    is_match = cellfun(predicate_function, all_class_properties) ;
    matching_class_properties = all_class_properties(is_match) ;
    property_names = cellfun(@(x)x.Name, matching_class_properties, 'UniformOutput', false) ;
end        
