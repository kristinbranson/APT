function result = get_home_dir_name()

if ispc() ,
  home = getenv('HOME') ;
  if ~isempty(home)
    result = getenv('HOME') ;
  else
    home_drive = getenv('HOMEDRIVE') ;
    home_path = getenv('HOMEPATH') ;  % will have backslashes as path separators
    result = horzcat(home_drive, home_path) ;
  end
else
  result = getenv('HOME') ;
end

end  % function
