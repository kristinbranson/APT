function testAPTci(varargin)
% Wrapper for testAPT.CIsuite
%
% MATLAB syntax oddity: 
%
% > testAPT.CIsuite arg1 arg2     % parse error; just bc of class static meth
% > randomfcn arg1 arg2           % passes arg1, arg2 as chars

testAPT.CIsuite(varargin{:});
