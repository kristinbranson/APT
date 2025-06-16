function dump_struct(s)
    % Lists all fields within s, recursively.  Currently, will only search scalar substructs.
    dump_struct_helper(s, cell(1,0)) ;
end



function dump_struct_helper(s, path_so_far)
    field_names = fieldnames(s) ;
    for i = 1 : length(field_names) ,
        field_name = field_names{i} ;
        path = horzcat(path_so_far, {field_name}) ;
        path_as_string = string_from_path_as_list(path) ;
        % If the field's value is a struct, recurse
        value = s.(field_name) ;
        if isstruct(value) ,
            fprintf('%s\n', path_as_string) ;
            dump_struct_helper(value, path) ;
        else
            fprintf('%s: %s\n', path_as_string, strtrim(formattedDisplayText(value))) ;
        end
    end
end



function result = string_from_path_as_list(path)
    if isempty(path) ,
        result = '' ;
    elseif isscalar(path) ,
        result = path{1} ;
    else
        result = strcatg(path{1}, '.', string_from_path_as_list(path(2:end))) ;
    end
end
