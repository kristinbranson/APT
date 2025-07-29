function result = read_file_into_char_array(file_name)
    fid = fopen(file_name, 'rb') ;
    if fid<0 ,
        error('Unable to open file %s for reading', file_name) ;
    end
    cleaner = onCleanup(@()(fclose(fid))) ;
    pre_result = fread(fid, inf, 'char=>char') ;
    result = (pre_result(:))' ;  % force into a row vector
end
