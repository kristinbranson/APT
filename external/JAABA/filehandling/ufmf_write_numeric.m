function ufmf_write_numeric(fid, a)
  if ~(isempty(a) || isvector(a)) ,
    error('Unable to write array to .ufmf index---vectors only') ;
  end
  fwrite(fid, 'a', 'char') ;
  class_name = class(a) ;  
  data_type_code = data_type_char(class_name) ;
  fwrite(fid, data_type_code, 'char') ;      
  n = length(a) ;
  byte_count_per_element = bytes_per_element(a) ;
  byte_count = n * byte_count_per_element ;
  fwrite(fid, byte_count, 'uint32') ;
  n_written = fwrite(fid, a, class_name) ;      
  if n_written ~= n ,
    error('Only able to write %d of a requested %d elements of type %s to array in .ufmf', n_written, n, class_name) ;
  end
end


function result = data_type_char(class_name) 
  if isequal(class_name, 'double') ,
    result = 'd' ;
  elseif isequal(class_name, 'uint8') ,
    result = 'B' ;
  elseif isequal(class_name, 'int64') ,
    result = 'q' ;
  else
    error('Unable to write data of class %s to .ufmf index', class_name) ;
  end
end


function bytes = bytes_per_element(x)
    w = whos('x') ;
    bytes = w.bytes / numel(x) ;
end
