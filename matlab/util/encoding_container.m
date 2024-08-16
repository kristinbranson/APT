function result = encoding_container(class_name, encoding)
    % Create an "encoding container", a scalar struct with fields "class_name" and
    % "encoding".  Used for persisting Matlab data structures of many kinds.
    result = struct('class_name',{class_name}, 'encoding',{encoding}) ;
end
