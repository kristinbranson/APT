function result = is_an_encoding_container(thing)
    % Returns true iff thing is an "encoding container", a scalar struct with
    % fields "class_name" and "encoding".  Used for persisting Matlab data
    % structures of many kinds.

    result = isstruct(thing) && isscalar(thing) && isfield(thing,'class_name') && isfield(thing,'encoding') ;
end
