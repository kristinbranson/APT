function APT_deployed(varargin)
  % Entry point for compiled matlab exectuable.
  try  
    % Parse the command-line arguments
    [do_test, test_name, is_lbl_given, lbl_file_name] = parse_args(varargin) ;

    % Either run test code or call StartAPT()
    if do_test
      % Run test(s) and exit
      if strcmp(test_name, 'all')
        test_apt() ;
      else
        test_function_name = sprintf('test_%s', test_name) ;
        test_apt(test_function_name) ;
      end
    else
      % Normal launch, not testing
      if is_lbl_given
        StartAPT('projfile', lbl_file_name) ;
      else
        StartAPT() ;
      end
    end
  catch ME
    errordlg(getReport(ME,'extended','hyperlinks','off'))  
  end
end


function [do_test, test_name, is_lbl_given, lbl_file_name] = parse_args(args)
  % Parse command-line arguments
  n = numel(args) ;
  is_lbl_given = false ;
  lbl_file_name = '' ;
  do_test = false ;
  test_name = '' ;
  just_saw_test_switch = false ;  % True iff last arg was '--test'
  for i = 1 : n
    arg = args{i} ;
    if just_saw_test_switch ,
      do_test = true ;
      test_name = arg ;
      just_saw_test_switch = false ;
    else
      if strcmp(arg, '--test')
        if do_test ,
          error('Only one test name is supported.  Use "all" to run all tests.')
        else
          just_saw_test_switch = true ;
        end
      else
        if is_lbl_given 
          error('Argument "%s" seem to a project file name, but one was already provided ("%s")', arg, lbl_file_name) ;
        else
          % arg must be .lbl name
          is_lbl_given = true ;
          lbl_file_name = arg ;
        end  % if
      end  % if 
    end  % if
  end  % for
  if just_saw_test_switch ,
    error('No test name given.  (Use "all" to run all tests.)')
  end
end  % function
