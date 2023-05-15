function command_string = synthesize_conda_command(conda_args_as_string)
% Synthesize a conda command string, trying to be smart about Linux/Windows, and exactly how conda
% was installed.  The intent is that this command string will be passed to
% system().

apt_conda_path = '/opt/conda/condabin/conda' ;
apt_conda_init_script_path = '/opt/conda/etc/profile.d/conda.sh' ;
if exist(apt_conda_path, 'file') && exist(apt_conda_init_script_path, 'file') ,
  % On Ubuntu, if you install conda via apt (apt as in Ubuntu/Debian apt-get,
  % not APT as in Adavanced Part Tracker), these files will exist, and we use
  % them as they are designed to be used
  % Editorial: On Ubuntu, you should install conda via apt.
  command_string = sprintf('source %s && conda %s', apt_conda_init_script_path, conda_args_as_string) ;
else
  % Just try plain old conda.  This might work...
  command_string = sprintf('conda %s', conda_args_as_string) ;
end

end
