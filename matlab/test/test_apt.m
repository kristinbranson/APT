function varargout = test_apt(varargin)
  % Run all the tests, except thos needing the AWS backend.  If keyword argument
  % 'aws' is true, the tests that use the AWS backend are also run.
  %
  % If a single argument is given, it is assumed to be a
  % single test function name, and it is feval'ed without a surrounding
  % try-catch block.  This is useful for debugging.  

  if numel(varargin)==1 ,
    test_count = 1 ;
    test_function_name = varargin{1} ;
    % Run the named test, without the try-catch block.  This is normally used for
    % debugging.
    fprintf('Running single test %s...\n', test_function_name) ;
    feval(test_function_name) ;
    % If get here, test passed.
    fprintf('Single test passed.\n') ;
    test_passed_count = 1 ;
  else
    [do_run_aws_tests] = myparse(varargin, ...
                                 'aws', false) ;
    this_dir_path = fileparts(mfilename('fullpath')) ;
    single_tests_dir_path = fullfile(this_dir_path, 'single-tests') ;
    test_file_names = simple_dir(fullfile(single_tests_dir_path, '*.m')) ;
    all_test_function_names = cellfun(@(file_name)(file_name(1:end-2)), test_file_names, 'UniformOutput', false) ;
    if do_run_aws_tests ,
      function_name_from_test_index = all_test_function_names ;
    else
      % Filter out tests with 'AWS' in the name, ignoring case
      is_aws_from_all_test_index = contains(all_test_function_names, 'aws', 'IgnoreCase', true) ;
      function_name_from_test_index = all_test_function_names(~is_aws_from_all_test_index) ;
    end
    test_count = numel(function_name_from_test_index) ;
    fprintf('Running %d tests...\n', test_count) ;
    did_pass_from_test_index = false(test_count,1) ;
    for test_index = 1 : test_count ,
      test_function_name = function_name_from_test_index{test_index} ;
      fprintf('\n\n\n\n\nRunning test %s...\n', test_function_name) ;
      try
        feval(test_function_name) ,
        did_pass_from_test_index(test_index) = true ;
        fprintf('Test %s (%d/%d) passed.\n', test_function_name, test_index, test_count) ;
      catch me
        fprintf('Test %s (%d/%d) failed:\n%s\n', test_function_name, test_index, test_count, me.getReport()) ;
      end
    end
    test_passed_count = sum(double(did_pass_from_test_index)) ;
    if test_passed_count == test_count ,
      fprintf('All tests (%d/%d) passed.\n', test_passed_count, test_count) ;
    else
      fprintf('Some tests failed: %d of %d tests passed.\n', test_passed_count, test_count) ;
      for test_index = 1 : test_count ,
        did_pass = did_pass_from_test_index(test_index) ;
        if ~did_pass ,
          function_name = function_name_from_test_index{test_index} ;
          fprintf('Test %s failed.\n', function_name) ;
        end
      end
    end
  end
  % Populate whatever return variables were requested
  varargout = cell(1,nargout) ;
  if nargout >=1 ,
    varargout{1} = test_count ;
  end
  if nargout >=2 ,
    varargout{2} = test_passed_count ;
  end
end
