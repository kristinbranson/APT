function find_in_struct(s, query_field_name)
    % Finds all fields named field_name within the scalar stuct s, and prints the
    % 'path' to them.  Currently, will only search scalar substructs.
    find_in_struct_helper(s, query_field_name, cell(1,0)) ;
end



function find_in_struct_helper(s, query_field_name, path_so_far)
    field_names = fieldnames(s) ;
    for i = 1 : length(field_names) ,
        field_name = field_names{i} ;
        path = horzcat(path_so_far, {field_name}) ;
        if contains(field_name, query_field_name, 'IgnoreCase', true) ,
            path_as_string = string_from_path_as_list(path) ;
            fprintf('%s\n', path_as_string) ;
        end
        % If the field's value is a struct, recurse
        value = s.(field_name) ;
        if isstruct(value) ,
            find_in_struct_helper(value, query_field_name, path) ;
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
