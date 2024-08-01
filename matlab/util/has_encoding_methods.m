function result = has_encoding_methods(object)
    % Tests for whether object has methods designed to ease encoding/decoding
    result = isobject(object) && ~isenum(object) && ...
             ( ismethod(object, 'encode_for_persistence_') || ...
               ( ismethod(object, 'get_property_value_') && ismethod(object, 'set_property_value_') ) ) ;
    % If there was an efficient way to check for a zero-arg constructor, we would
    % do that here also.  
end
