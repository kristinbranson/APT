function status_string = ...
  interpolate_shorter_status_string(raw_status_string, hasProject, projectfile)

% The status string kept in the Labeler object can have the string
% '$PROJECTNAME' in it.  In this case, the actual project file *path* is
% substituted for '$PROJECTNAME' before being displayed in the GUI.  The
% function interpolate_status_string() performs this substitution (see the
% help text for that function).  If the result is too long, the
% LabelerController invokes this function to get a shorter version.
%
% The status string kept in the Labeler object can have the string
% '$PROJECTNAME' in it.  In this case, this function substitutes the  actual
% project file *name* for '$PROJECTNAME'.  Additionally, the project file name
% is trimmed to 100 characters if needed, with a leading '...'.

does_input_string_contain_variable = contains(raw_status_string,'$PROJECTNAME') ;
if does_input_string_contain_variable && hasProject ,
  if ~ischar(projectfile) ,
    project_file_path = '' ;
  else
    project_file_path = projectfile ;
  end
  max_length = 100 ;
  [~,project_file_name] = fileparts2(project_file_path) ;
  trimmed_project_file_name = trim_string_to_length(project_file_name, max_length) ;
  status_string = strrep(raw_status_string,'$PROJECTNAME',trimmed_project_file_name) ;
else
  status_string = raw_status_string ;
end

end
