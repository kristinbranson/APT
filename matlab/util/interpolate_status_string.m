function status_string = ...
  interpolate_status_string(raw_status_string, hasProject, projectfile)

% The status string kept in the Labeler object can have the string
% '$PROJECTNAME' in it.  In this case, the actual project file path is
% substituted for '$PROJECTNAME' before being displayed in the GUI.  This
% function performs this substitution.  Additionally, the project file path is
% trimmed to 100 characters if needed, with a leading '...'.  Note that
% depending on the length of the result, the LabelerController may decide to
% use the function interpolate_shorter_status_string() to get a shorter
% version (see the help text for that function for more information).

does_input_string_contain_variable = contains(raw_status_string,'$PROJECTNAME') ;
if does_input_string_contain_variable && hasProject ,
  if ~ischar(projectfile) || isempty(projectfile) ,
    project_file_path = '<unnamed project>' ;
  else
    project_file_path = projectfile ;
  end
  max_length = 100 ;
  trimmed_project_file_path = trim_string_to_length(project_file_path, max_length) ;
  status_string = strrep(raw_status_string,'$PROJECTNAME',trimmed_project_file_path) ;
else
  status_string = raw_status_string ;
end

end
