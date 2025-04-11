function result = wrapCommandConda(cmd, varargin)
% Take a base command and run it in a conda env

% Process keyword arguments
[condaEnv,gpuid,logfile] = ...
  myparse(varargin,...
          'condaEnv',[],...
          'gpuid',0, ...
          'logfile','');
if isempty(condaEnv) ,
  error('condaEnv argument is mandatory') ;
end

% Find the conda executable
conda_exectuable_path = find_conda_executable() ;

% Augment the command with a specification of the GPU id, if called for
if isempty(gpuid) || isnan(gpuid) ,
  partial_command = cmd ;
else
  partial_command = sprintf('export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES=%d && %s', gpuid, cmd) ;
end

% Add logging
if isempty(logfile) ,
  full_command = partial_command ;
else
  quoted_log_file_name = escape_string_for_bash(logfile) ;
  full_command = sprintf('%s &> %s', partial_command, quoted_log_file_name) ;
end

% Quote the command for bash
quoted_full_command = escape_string_for_bash(full_command) ;

% The command will use the conda "run" subcommand, and use bash explicitly to
% interpret the command line
preresult = sprintf('%s run -n %s /bin/bash -c %s', ...
                    conda_exectuable_path, ...
                    condaEnv, ...
                    quoted_full_command) ;

% Clear annoying Matlab envars
result = prepend_stuff_to_clear_matlab_environment(preresult) ;
