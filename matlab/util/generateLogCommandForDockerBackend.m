function cmd = generateLogCommandForDockerBackend(backend, containerName, native_log_file_name)

assert(backend.type == DLBackEnd.Docker);
dockercmd = apt.dockercmd();
log_file_name = linux_path(native_log_file_name) ;
cmd = ...
  sprintf('%s logs -f %s &> %s', ... 
          dockercmd, ...
          containerName, ...
          escape_string_for_bash(log_file_name)) ;
is_docker_remote = ~isempty(obj.dockerremotehost) ;
if is_docker_remote
  cmd = wrapCommandSSH(cmd,'host',obj.dockerremotehost);
end
cmd = [cmd,' &'];
