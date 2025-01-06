function result = test(varargin)
    % apt.test()  Run APTwavesurfer automated tests.
    %
    %   apt.test() runs all APT automated tests.
    %
    %   res = apt.test() returns test results in the result
    %   structure, res, rather than displaying the results at the command
    %   line.

    % By default include the tests that don't require hardware.
    testSuite = matlab.unittest.TestSuite.fromPackage('apt.test');

    % Deal with duplicate tests, which happens sometimes, for unknown
    % reasons
    test_names = {testSuite.Name} ;
    [unique_test_names, indices_of_unique_tests] = unique(test_names) ;  %#ok<ASGLU>
    testSuiteWithAllUnique = testSuite(indices_of_unique_tests) ;
    if length(testSuiteWithAllUnique) ~= length(testSuite) ,
        warning('Sigh.  There seem to be duplicated tests...') ;
    end
    
    % Run the (unique) tests
    fprintf('About to perform %d tests...\n', length(testSuiteWithAllUnique)) ;
    result = testSuiteWithAllUnique.run() ;
end
