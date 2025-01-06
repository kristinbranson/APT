function jobid = parseJobID(backend, res)

switch backend.type
  case DLBackEnd.AWS,
    jobid = apt.parseJobIDAWS(res) ;
  case DLBackEnd.Bsub,
    jobid = apt.parseJobIDBsub(res) ;
  case DLBackEnd.Conda,
    jobid = apt.parseJobIDConda(res) ;
  case DLBackEnd.Docker,
    jobid = apt.parseJobIDDocker(res) ;
  otherwise
    error('Not implemented: %s',backend.type);
end
