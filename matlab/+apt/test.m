function result = test(varargin)
    % apt.test()  Run APT automated tests.
    %
    %   apt.test() runs all APT automated tests.
    %
    %   res = apt.test() returns test results in the result
    %   structure, res, rather than displaying the results at the command
    %   line.

    % By default include the tests that don't require AWS
    noAWSTestSuite = matlab.unittest.TestSuite.fromPackage('ws.test.noaws');

    % Add the hardware tests if appropriate based on the input arguments.
    if any(strcmp('--aws', varargin)) ,
        % Add the AWS tests if requested
        withAWSTestSuite = matlab.unittest.TestSuite.fromPackage('ws.test.aws');
        testSuite = horzcat(withAWSTestSuite, noAWSTestSuite) ;
    else
        % Just the no-AWS tests 
        testSuite = noAWSTestSuite ;
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
