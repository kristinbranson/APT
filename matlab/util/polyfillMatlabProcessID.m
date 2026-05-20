function pid = polyfillMatlabProcessID()
% Return the OS process ID of the current MATLAB process.  Uses the
% matlabProcessID() built-in when available (MATLAB 2026a and newer);
% otherwise falls back to the undocumented feature('getpid') call.
persistent useBuiltin
if isempty(useBuiltin)
  useBuiltin = logical(exist('matlabProcessID', 'builtin')) || ...
               logical(exist('matlabProcessID', 'file')) ;
end
if useBuiltin
  pid = matlabProcessID() ;
else
  pid = feature('getpid') ;
end
end  % function
