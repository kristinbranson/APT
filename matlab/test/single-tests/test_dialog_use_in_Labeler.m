function test_dialog_use_in_Labeler()
  % Test for calls to questdlg(), warndlg() in Labeler.
  % All such calls should eventually be moved out of Labeler.m, likely into
  % LabelerController.m.  This test makes sure we don't backslide.
  script_path = mfilename('fullpath') ;
  apt_path = fileparts(fileparts(fileparts(fileparts(script_path)))) ;
  labeler_source_file_path = fullfile(apt_path, 'matlab/Labeler.m') ;
  labeler_source_lines = read_file_into_cellstring(labeler_source_file_path) ;
  % Check for calls to questdlg()
  does_contain_questdlg_from_line_index = cellfun(@(line)(contains(line, 'questdlg(')), labeler_source_lines) ;
  questdlg_line_count = sum(does_contain_questdlg_from_line_index) ;
  if questdlg_line_count > 9
    error('At most 9 calls to questdlg() are allowed in Labeler.m, but it looks like there are %d now', questdlg_line_count) ;
  end
  % Check for calls to warndlg()
  does_contain_warndlg_from_line_index = cellfun(@(line)(contains(line, 'warndlg(')), labeler_source_lines) ;
  warndlg_line_count = sum(does_contain_warndlg_from_line_index) ;
  if warndlg_line_count > 2
    error('At most 2 calls to warndlg() are allowed in Labeler.m, but it looks like there are %d now', warndlg_line_count) ;
  end
end  % function
