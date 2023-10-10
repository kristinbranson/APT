function result = prepend_stuff_to_clear_matlab_environment(input_command_line) 
% When you call system(), the envars are polluted with a bunch of 
% Matlab-specific things that can break stuff.  E.g. Matlab changes
% LD_LIBRARY_PATH, and that often breaks code you'd like to run with system().
% The prepends a bunch of unset and export commands to your command line to
% fix these issues.

% Matlab sets all these envars, at least one of which seem to cause the PyTorch
% dataloader to segfault.  So we unset them all.
envar_names_to_clear = ...
  { 'ARCH', 'AUTOMOUNT_MAP', 'BASEMATLABPATH', 'ICU_TIMEZONE_FILES_DIR', 'KMP_BLOCKTIME', 'KMP_HANDLE_SIGNALS', 'KMP_INIT_AT_FORK', ...
    'KMP_STACKSIZE', 'LC_NUMERIC', 'LD_PRELOAD', 'LIBVA_MESSAGING_LEVEL', 'MEMKIND_HEAP_MANAGER', 'MKL_DOMAIN_NUM_THREADS', ...
    'MKL_NUM_THREADS', 'OSG_LD_LIBRARY_PATH', 'PRE_LD_PRELOAD', 'TOOLBOX', 'XFILESEARCHPATH' } ;
unset_commands = cellfun(@(str)(sprintf('unset %s', str)), envar_names_to_clear, 'UniformOutput', false) ;
unset_command_line = strjoin(unset_commands, ' && ') ;

% We want to parse the LD_LIBRARY_PATH and purge it of any Matlab-related
% stuff.  If it's not set we can skip this.
if isenv('LD_LIBRARY_PATH') 
  orginal_ld_library_path = getenv('LD_LIBRARY_PATH') ;
  dir_from_original_path_index = strsplit(orginal_ld_library_path, ':') ;
  is_matlaby_from_original_path_index = ...
      contains(dir_from_original_path_index, 'matlab', 'IgnoreCase', true) | contains(dir_from_original_path_index, 'mathworks', 'IgnoreCase', true) ;
  dir_from_path_index = dir_from_original_path_index(~is_matlaby_from_original_path_index) ;
  ld_library_path = strjoin(dir_from_path_index, ':') ;
  ld_library_path_export_command = horzcat('export LD_LIBRARY_PATH=', ld_library_path) ;
  % Join all the sub-commands with &&.  Deal gracefully with an empty
  % input_command_line.
  if isempty(input_command_line) ,
    result = horzcat(unset_command_line, ' && ', ld_library_path_export_command) ;
  else
    result = horzcat(unset_command_line, ' && ', ld_library_path_export_command, ' && ', input_command_line) ;
  end
else
  % Join all the sub-commands with &&.  Deal gracefully with an empty
  % input_command_line.
  if isempty(input_command_line) ,
    result = unset_command_line ;
  else    
    result = horzcat(unset_command_line, ' && ', input_command_line) ;
  end
end

end