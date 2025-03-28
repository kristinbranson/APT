function build()
  % Build a StartAPT executable.
  this_script_path = mfilename('fullpath') ;
  apt_root_folder = fileparts(this_script_path) ;  % Should be the project root
  executable_folder_path = fullfile(apt_root_folder, 'executable') ;
  mcc('-v', '-m', 'StartAPT.m', '-d', executable_folder_path, '-a', fullfile(apt_root_folder, 'matlab', 'createLabelerMainFigure_assets.mat')) ;
end
