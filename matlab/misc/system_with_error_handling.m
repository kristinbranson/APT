function stdout = system_with_error_handling(command_line)
    [return_code, stdout] = system(command_line) ;
    if return_code ~= 0 ,
        error('system_with_error_handling:nonzero_return_code', ...
              'There was a problem running the command:\n%s\nReturn code: %d\nStdout/stderr:\n%s', command_line, return_code, stdout) ;
    end
end
