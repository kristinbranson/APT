function retval=isFileNameAbsolute(fileName)

% If you do x=fileparts(x) until you reach steady-state,
% the steady-state x will be empty if and only if the initial x is relative.
% If absolute, the steady-state x will be "/" on Unix-like OSes, and 
% something like "C:\" on Windows.
%
% Note that this will return true for the empty string.  This may be
% convoversial.  But, you know: garbage in, garbage out.

path=fileName;
parent=fileparts(path);
while ~strcmp(path,parent)
  path=parent;
  parent=fileparts(path);
end
% at this point path==parent
retval=~isempty(path);

end
