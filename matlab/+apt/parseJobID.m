function jobID = parseJobID(backend, res)

switch backend.type
  case DLBackEnd.AWS,
    jobID = apt.parseJobIDAWS(res) ;
  case DLBackEnd.Bsub,
    jobID = apt.parseJobIDBsub(res) ;
  case DLBackEnd.Conda,
    jobID = apt.parseJobIDConda(res) ;
  case DLBackEnd.Docker,
    jobID = apt.parseJobIDDocker(res) ;
  otherwise
    error('Not implemented: %s',backend.type);
end
