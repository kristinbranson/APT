function result = wrap_linux_command_line_for_wsl_if_needed(linux_command_line)

if ispc() ,
  escaped_linux_command_line = escape_string_for_command_dot_exe(linux_command_line) ;
  result = sprintf('wsl -- bash -c %s', escaped_linux_command_line) ;
else
  result = linux_command_line ;
end
