function result = wrap_linux_command_line_for_wsl(cmd)

escaped_cmd =  escape_string_for_command_dot_exe(cmd) ;
result = ['wsl -- bash -c ', escaped_cmd] ;

end
