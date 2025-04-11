function clusterstr = monitorBackendDescription(backend_enum)
  clusterstr = 'Cluster';
  switch backend_enum        
    case DLBackEnd.Bsub
      clusterstr = 'JRC cluster';
    case DLBackEnd.Conda
      clusterstr = 'Local';
    case DLBackEnd.Docker
      clusterstr = 'Local';
    case DLBackEnd.AWS,
      clusterstr = 'AWS';
    otherwise
      warning('Unknown back end type');
  end
end  % function
