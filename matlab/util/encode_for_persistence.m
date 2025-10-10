function result = encode_for_persistence(thing, do_wrap_in_container)
    % Encode any Matlab data structure in a struct for persisting.  If
    % do_wrap_in_container is true, wraps the structure in an "encoding
    % container", a scalar struct with fields "class_name" and "encoding", so that
    % the class can be recreated after the encoding container is read from disk.
    % If missing or empty, do_wrap_in_container defaults to true.

    % Deal with args
    if ~exist('do_wrap_in_container', 'var') || isempty(do_wrap_in_container) ,
        do_wrap_in_container = true ;
    end

    if isnumeric(thing) || ischar(thing) || islogical(thing) || isa(thing, 'function_handle') ,
        % These build-in things are their own encoding
        result = maybe_wrap_in_encoding_container(thing, thing, do_wrap_in_container) ;
    elseif iscell(thing) ,
        encoding=cell(size(thing));
        for j=1:numel(thing) ,
            encoding{j} = encode_for_persistence(thing{j}, do_wrap_in_container);
        end
        result = maybe_wrap_in_encoding_container(encoding, thing, do_wrap_in_container) ;
    elseif isstruct(thing) ,
        fieldNames=fieldnames(thing);
        encoding=ws.structWithDims(size(thing),fieldNames);
        for i=1:numel(thing) ,
            for j=1:length(fieldNames) ,
                thisFieldName=fieldNames{j};
                encoding(i).(thisFieldName) = ws.encodeAnythingForPersistence(thing(i).(thisFieldName)) ;
            end
        end
        result = maybe_wrap_in_encoding_container(encoding, thing, do_wrap_in_container) ;
    elseif isenum(thing) ,
        encoding = char(thing) ;
        result = maybe_wrap_in_encoding_container(encoding, thing, do_wrap_in_container) ;        
    elseif has_encoding_methods(thing) ,
        if ismethod(thing, 'encode_for_persistence_') ,
            result = thing.encode_for_persistence_(do_wrap_in_container) ;
        else
            result = encode_object_for_persistence(thing, do_wrap_in_container) ;
        end
    else                
        error('APT:dont_know_how_to_encode', ...
              'Don''t know how to encode an entity of class %s', ...
              class(thing));
    end
end  % function                
