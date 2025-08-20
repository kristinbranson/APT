function result = read_file_into_cellstring(file_name)
    % Read a text file into a cellstring, one line per element.
    % Newlines are discarded.
    file = read_file_into_char_array(file_name) ;
    % Delete carriage returns, in case it's a Windows-style file
    is_cr = (file==char(13)) ;
    file(is_cr) = [] ;
    % If the last char is a newline, get rid of it
    nl = newline() ;  % == char(10)
    if file(end) == nl ,
      file(end) = [] ;
    end
    % Finally, split at newlines
    result = strsplit(file, nl) ;
end




% This version is sloooow:    
%     fid = fopen(file_name, 'rt') ;
%     if fid<0 ,
%         error('read_file_into_cell_string:unable_to_open_file', 'Unable to open file %s for reading', file_name) ;
%     end
%     cleaner = onCleanup(@()(fclose(fid))) ;
%     result = cell(0,1) ;
%     line = fgetl(fid) ;
%     while ischar(line) ,
%         result = vertcat(result, ...
%                          {line}) ;  %#ok<AGROW>
%         line = fgetl(fid) ;
%     end
