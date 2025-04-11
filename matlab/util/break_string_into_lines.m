function result = break_string_into_lines(str)
    % Break a string at newlines into a cellstring of lines.
    % Newlines are discarded.

    % Delete carriage returns, in case it's a Windows-style file
    is_cr = (str==char(13)) ;
    str(is_cr) = [] ;
    % If the last char is a newline, get rid of it
    nl = newline() ;  % == char(10)
    if str(end) == nl ,
      str(end) = [] ;
    end
    % Finally, split at newlines
    result = strsplit(str, nl) ;
end
