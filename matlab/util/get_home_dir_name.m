function result = get_home_dir_name()

if ispc() ,
  home_drive = getenv('HOMEDRIVE') ;
  home_path = getenv('HOMEPATH') ;  % will have backslashes as path separators
  result = horzcat(home_drive, home_path) ;
else
  result = getenv('HOME') ;
end  

end
