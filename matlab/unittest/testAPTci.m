function testAPTci(varargin)
% Wrapper for TestAPT.runCITestSuite
%
% MATLAB syntax oddity: 
%
% > TestAPT.runCITestSuite arg1 arg2     % parse error; just bc of class static meth
% > randomfcn arg1 arg2           % passes arg1, arg2 as chars

TestAPT.runCITestSuite(varargin{:});
