function stdout = system_with_error_handling(command_line)
    [return_code, stdout] = system(command_line) ;
    if return_code ~= 0 ,
        error('There was a problem running the command "%s".  Return code: %d.  Stdout/stderr:\n%s', command_line, return_code, stdout) ;
    end
end
