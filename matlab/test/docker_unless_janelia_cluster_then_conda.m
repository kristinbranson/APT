function result = docker_unless_janelia_cluster_then_conda()
  lsb_exec_cluster = getenv('LSB_EXEC_CLUSTER') ;
  if strcmpi(lsb_exec_cluster, 'Janelia')
    result = 'conda' ;
  else
    result = 'docker' ;
  end
end
