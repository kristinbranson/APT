function status_string = ...
  interpolate_shorter_status_string(raw_status_string, hasProject, projectfile)

does_input_string_contain_variable = contains(raw_status_string,'$PROJECTNAME') ;
if does_input_string_contain_variable && hasProject ,
  if ~ischar(projectfile) ,
    project_file_path = '' ;
  else
    project_file_path = self.projectfile ;
  end
  max_length = 100 ;
  [~,project_file_name] = fileparts2(project_file_path) ;
  trimmed_project_file_name = trim_string_to_length(project_file_name, max_length) ;
  status_string = strrep(raw_status_string,'$PROJECTNAME',trimmed_project_file_name) ;
else
  status_string = raw_status_string ;
end

end
