function result = decode_encoding_container(encoding_container, warning_logger)
    if nargin<2 ,
        warning_logger = [] ;
    end
    % Unpack the encoding container, or try to deal with it if
    % encodingContainer is not actually an encoding container.            
    if is_an_encoding_container(encoding_container) ,
        % Unpack the fields of the encodingContainer
        class_name = encoding_container.class_name ;
        encoding = encoding_container.encoding ;
    else
        error('APT:not_an_encoding_container', 'Can only decode an encoding container') ;
    end

    % Create the object to be returned
    if is_a_numeric_class_name(class_name) || ...
       strcmp(class_name,'char') || ...
       strcmp(class_name,'logical') || ...
       strcmp(class_name, 'function_handle') ,
        result = encoding ;
    elseif isequal(class_name,'cell') ,
        result = cell(size(encoding)) ;
        for i=1:numel(result) ,
            result{i} = decode_encoding_container(encoding{i}, warning_logger) ;
        end
    elseif isequal(class_name,'struct') ,
        field_names = fieldnames(encoding) ;
        result = struct_with_dims(size(encoding), field_names) ;
        for i=1:numel(encoding) ,
            for j=1:length(field_names) ,
                field_name = field_names{j} ;
                result(i).(field_name) = decode_encoding_container(encoding(i).(field_name), warning_logger) ;
            end
        end
    elseif isenum_from_class_name(class_name) ,
        result = feval(class_name, encoding) ;
    else
        % We assume at this point that the encoded thing is a member of a "model"
        % class.  By this, we mean a class with a zero-arg constructor, which has
        % public get_property_value_() and set_property_value_() methods.

        % Make sure the encoded object is a scalar
        if isscalar(encoding) ,
            % Check for a custom class method
            if does_class_have_static_method_named('decode_encoding', class_name) ,
                full_static_method_name = horzcat(class_name, '.decode_encoding') ;
                result = feval(full_static_method_name, encoding) ;
            else
                % Instantiate the object
                result = feval(class_name) ;  % We assume a zero-arg constructor here

                % Get the list of persistable properties
                if ismethod(result, 'list_properties_for_persistence') ,
                    persisted_property_names = result.list_properties_for_persistence() ;
                else
                    persisted_property_names = list_properties_for_persistence(result) ;
                end

                % Get the property names from the encoding
                field_names = fieldnames(encoding) ;

                % Set each property name in self
                for i = 1:numel(field_names) ,
                    field_name = field_names{i};
                    % Usually, the property_name is the same as the field
                    % name, but we do some ad-hoc translations to support
                    % old files.
                    if ~isprop(result,field_name) || ~ismember(field_name, persisted_property_names) ,
                        % If the field name is not a property name or the object, or not a persisted
                        % property name, try adding an underscore to the field name.  (But first check
                        % that the underscored version is not already a field in the encoding.  And
                        % don't do this if the field name already ends in an underscore.)
                        if ~isempty(field_name) && ~strcmp(field_name(end),'_') 
                            underscored_field_name = strcatg(field_name, '_') ;
                            if ismember(underscored_field_name, field_names) ,
                                % The underscored name is already a field name, so just use the original field
                                % name, which will just generate a warning below.
                                property_name = field_name ;
                            else
                                % The underscored name is not already a field name, so check if the
                                % underscored version is a persisted property name.
                                if isprop(result,underscored_field_name) && ismember(underscored_field_name, persisted_property_names)
                                    % Go ahead and use the underscored version as the property name
                                    property_name = underscored_field_name ;
                                else
                                    % Just use the original field name, which will just generate a warning below.
                                    property_name = field_name ;
                                end
                            end
                        else
                          % The field name already ends with an underscore, do just leave it alone and
                          % let it generate a warning below.
                          property_name = field_name ;
                        end
                    else
                        % The typical case
                        property_name = field_name ;
                    end
                    % Only decode if there's a property to receive it, and that property is one of
                    % the persisted ones.
                    if isprop(result,property_name) && ismember(property_name, persisted_property_names) ,
                        subencoding = encoding.(field_name) ;  % the encoding is a struct, so no worries about access
                        if false ,
                            % Put backwards-compatibility hacks here.
                        else
                            % the usual case
                            do_set_property_value = true ;
                            subresult = decode_encoding_container(subencoding, warning_logger) ;
                        end
                        if do_set_property_value ,
                            try
                                result.set_property_value_(property_name, subresult) ;
                            catch me
                                if ~isempty(warning_logger) ,
                                    warning_logger.log_warning('APT:error_setting_property', ...
                                        sprintf('Ignoring error when attempting to set property %s a thing of class %s: %s', ...
                                        property_name, ...
                                        class_name, ...
                                        me.message), ...
                                        me) ;
                                end
                            end
                        end
                    else
                        if ~isempty(warning_logger) ,
                            warning_logger.log_warning('APT:error_setting_property', ...
                                sprintf('Ignoring field ''%s'' from the file, because the corresponding property %s is not present in the %s object.', ...
                                field_name, ...
                                property_name, ...
                                class(result))) ;
                        end
                    end
                end  % for over field_names

                % Do sanity-checking on persisted state
                if ismethod(result, 'sanitize_persisted_state_') ,
                    result.sanitize_persisted_state_() ;
                end

                % Make sure the transient state is consistent with
                % the non-transient state
                if ismethod(result, 'synchronize_transient_state_to_persisted_state_') ,
                    result.synchronize_transient_state_to_persisted_state_() ;
                end
            end
        else
            % The encoding is not a scalar, which it should be.                    
            % Again, we'd be within our rights to throw an error
            % here, but we try to be helpful...
            n = numel(encoding) ;
            result = cell(1,n) ;
            for i=1:n ,
                hacked_container = struct('class_name', class_name, 'encoding', encoding(i)) ;
                result{i} = decode_encoding_container(hacked_container, warning_logger) ;
            end
        end
    end
end
