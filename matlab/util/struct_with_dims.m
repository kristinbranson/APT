function result=struct_with_dims(dims, field_names)
    % Creates a structure with the desired dimensions and field names.    
    % dims a row vector of dimensions, field_names a cell array of strings
    
    if length(dims)==0 , %#ok<ISMT>
        dims=[1 1];
    elseif length(dims)==1 ,
        dims=[dims 1];
    end
    
    if isempty(field_names) ,
        was_field_names_empty=true;
        field_names={'foo'};
    else
        was_field_names_empty=false;
    end
        
    template=cell(dims);
    nFields=length(field_names);
    args=cell(1,2*nFields);
    for fieldIndex=1:nFields ,
        argIndex1=2*(fieldIndex-1)+1;
        argIndex2=argIndex1+1;
        args{argIndex1}=field_names{fieldIndex};
        args{argIndex2}=template;  
    end
    
    result=struct(args{:});
    
    if (was_field_names_empty) ,
        result=rmfield(result,'foo');
    end    
end
