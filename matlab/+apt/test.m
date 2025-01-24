function result = test(varargin)
    % apt.test()  Run APT automated tests.
    %
    %   apt.test() runs all APT automated tests, except the AWS ones.
    %   apt.test('--aws') runs all APT automated tests, including the AWS ones.
    %
    %   res = apt.test() returns test results in the result
    %   structure, res, rather than displaying the results at the command
    %   line.

    % By default include the tests that don't require AWS
    coreTestSuite = matlab.unittest.TestSuite.fromPackage('apt.test');

    % Add the AWS tests if appropriate based on the input arguments.
    if any(strcmp('--aws', varargin)) ,
        % Add the AWS tests if requested
        withAWSTestSuite = matlab.unittest.TestSuite.fromPackage('apt.test.aws');
        testSuite = horzcat(withAWSTestSuite, coreTestSuite) ;
    else
        % Just the no-AWS tests 
        testSuite = coreTestSuite ;
    end

    % Deal with duplicate tests, which happens sometimes, for unknown
    % reasons
    test_names = {testSuite.Name} ;
    [unique_test_names, indices_of_unique_tests] = unique(test_names) ;  %#ok<ASGLU>
    testSuiteWithAllUnique = testSuite(indices_of_unique_tests) ;
    if length(testSuiteWithAllUnique) ~= length(testSuite) ,
        warning('Sigh.  There seem to be duplicated tests...') ;
    end
    
    % Run the tests
    fprintf('About to perform %d tests...\n', length(testSuiteWithAllUnique)) ;
    result = testSuiteWithAllUnique.run() ;
end
