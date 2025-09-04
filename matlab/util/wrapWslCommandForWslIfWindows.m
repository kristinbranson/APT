function result = wrapWslCommandForWslIfWindows(baseCommand)

assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be a ShellCommand') ;
assert(baseCommand.locale_ == apt.PathLocale.wsl, 'baseCommand must have WSL locale') ;

if ispc() ,
  result = apt.ShellCommand({'wsl', '--exec', 'bash', '-c', baseCommand}, apt.PathLocale.native, apt.Platform.windows) ;
    % The --exec here is critically important.  See here:
    %   https://github.com/microsoft/WSL/issues/1746#issuecomment-650347125
    %
    % Some examples:
    % C:\Users\taylora>wsl bash -c "echo foo && false ; echo $?"
    % foo
    % 0   [<-- Wrong!  The double-quoted part is seemingly getting passed to bash as a double-quoted string, so variable interpolation happens too early.]
    % 
    % C:\Users\taylora>wsl --exec bash -c "echo foo && false ; echo $?"
    % foo
    % 1   [<--- Correct!  The double-quoted part is parsed by command.exe as a single token, and seemingly passed to bash as such.]
    % 
    % C:\Users\taylora>wsl --exec bash -c 'echo foo && false ; echo $?'
    % foo: -c: line 1: unexpected EOF while looking for matching `''  
    % foo: -c: line 2: syntax error: unexpected end of file
    %  [ Bad.  The && is getting interpreted by command.exe, despite the single
    %    quotes. ]
    % 
    % C:\Users\taylora>wsl bash -c 'echo foo && false ; echo $?'
    % /bin/bash: -c: line 1: unexpected EOF while looking for matching `''
    % /bin/bash: -c: line 2: syntax error: unexpected end of file
    %  [ Bad.  The && is getting interpreted by command.exe, despite the single
    %    quotes. ]
    % 
    % C:\Users\taylora>    
else
  result = baseCommand ;
end
