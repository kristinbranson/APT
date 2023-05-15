function result = encode_object_for_persistence(object, do_wrap_in_container)
    % Used to encode encodable objects.  I.e. objects for which
    % is_encodable_object() is true.

    % Deal with args
    if ~exist('do_wrap_in_container', 'var') || isempty(do_wrap_in_container) ,
        do_wrap_in_container = true ;
    end

    % object needs to be a scalar...
    if ~isscalar(object) ,
        error('APT:cant_encode_nonscalar', ...
              'Can''t encode a nonscalar object of class %s', ...
              class(object));
    end
    if ~ismethod(object, 'get_property_value_') ,
        error('APT:cant_encode_object_lacking_get_property_value__method', ...
              'Can''t encode an object that lacks a get_property_value_() method');
    end            

    % Get the list of property names for this file type
    if ismethod(object, 'list_properties_for_persistence') ,
        property_names = object.list_properties_for_persistence() ;
    else
        property_names = list_properties_for_persistence(object) ;
    end

    % Encode the value for each property
    encoding = struct() ;  % scalar struct with no fields
    for i = 1:length(property_names) ,
        property_name=property_names{i} ;
        property_value = object.get_property_value_(property_name) ;
        encoding_of_property_value = encode_for_persistence(property_value, do_wrap_in_container) ;
        encoding.(property_name) = encoding_of_property_value ;
    end

    % For restorable encodings, usually want to make an "encoding
    % container" that captures the class name.
    result = maybe_wrap_in_encoding_container(encoding, object, do_wrap_in_container) ;
end
