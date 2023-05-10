function status_string = ...
  interpolate_status_string(raw_status_string, hasProject, projectfile)

does_input_string_contain_variable = contains(raw_status_string,'$PROJECTNAME') ;
if does_input_string_contain_variable && hasProject ,
  if ~ischar(projectfile) ,
    project_file_path = '' ;
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
