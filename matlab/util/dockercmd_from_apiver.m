function s = dockercmd_from_apiver(apiver)

s = sprintf('export DOCKER_API_VERSION=%s ; docker',apiver) ;

end
