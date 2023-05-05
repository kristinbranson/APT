function result = maybe_wrap_in_encoding_container(encoding, object, do_wrap_in_container)
    % Wrap the encoding of the object_or_class_name in an "encoding container", or not,
    % depending on do_wrap_in_container.
    if do_wrap_in_container ,
        class_name = class(object) ;
        result = encoding_container(class_name, encoding) ;
    else
        result = encoding ;
    end
end
