function ufmf_write_struct(fid, s)
  if ~(isstruct(s) && isscalar(s)) ,
    error('Can''t write a non-scalar struct') ;
  end
  fwrite(fid, 'd', 'char') ;  
  field_names = fieldnames(s) ;
  field_count = length(field_names) ;
  fwrite(fid, uint16(field_count), 'uint8') ;  % seems small, but this is what ufmf_read_header() reads...
  for i = 1 : field_count ,
    % Write the field name
    field_name = field_names{i} ;
    fwrite(fid, length(field_name), 'uint16') ;
    fwrite(fid, field_name, 'char') ;    
    % Write the field value
    value = s.(field_name) ;
    if isstruct(value) ,
      ufmf_write_struct(fid, value) ;
    elseif isnumeric(value) ,
      ufmf_write_numeric(fid, value) ;
    else
      error('Unable to write entity of class %s to .ufmf index', classname(value)) ;
    end
  end
end


