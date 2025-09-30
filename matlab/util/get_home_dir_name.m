function result = get_home_dir_name()

if ispc() ,
  home = getenv('HOME') ;
  if ~isempty(home)
    result = home ;
  else
    result = getenv('USERPROFILE') ;  
      % Works better in practice, partly b/c Janelia managed PCs have
      % HOMEDRIVE="U:"
      % and HOMEDRIVE="\"
      % which means the APT cache ends up on a network share.
    % home_drive = getenv('HOMEDRIVE') ;
    % home_path = getenv('HOMEPATH') ;  % will have backslashes as path separators
    % result = horzcat(home_drive, home_path) ;
  end
else
  result = getenv('HOME') ;
end

end  % function
