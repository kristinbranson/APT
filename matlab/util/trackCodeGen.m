function codestr = trackCodeGen(backend, varargin)

switch backend.type ,
  case DLBackEnd.Docker,
    codestr = trackCodeGenDocker(backend, varargin{:});
  case DLBackEnd.AWS,
    codestr = trackCodeGenAWS(backend, varargin{:});
  case DLBackEnd.Bsub,
    codestr = trackCodeGenSSHBsubSing(backend, varargin{:});
  case DLBackEnd.Conda,
    codestr = trackCodeGenConda(backend, varargin{:});
  otherwise ,
    error('Unknown DLBackEnd type') ;
end  % switch

end  % function
