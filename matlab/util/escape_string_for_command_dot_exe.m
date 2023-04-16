function result = escape_string_for_command_dot_exe(str)
    % Process the string str so that when the result is passed as part of a
    % Windows command.exe command line, it will be interpreted as a single
    % token, and the string received by the accepting executable/script will be
    % identical to str.
    
    % Basically, we need to replace each occurance of " with \", and each
    % occurance of \ with \\.  Then we add a " to either end.

    % First replace the \'s with \\'s
    bs = '\' ;
    bsbs = '\\' ;
    preresult_1 = strrep(str, bs, bsbs) ;

    % Next replace the "'s with \"'s
    % (Have to do this second, or else we would de-escape the \'s we put in.)
    dq = '"' ;
    bsdq = '\"' ;
    preresult_2 = strrep(preresult_1, dq, bsdq) ;
    
    % Wrap in "'s
    result = horzcat(dq, preresult_2, dq) ;
end
