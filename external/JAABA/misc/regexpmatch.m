function tf = regexpmatch(s,pat,varargin)
% tf = regexpmatch(s,pat,p1,v1,...)
% Either s xor pat is expected to be a cellstr.
% Optional PVs:
% - caseinsens, logical scalar

assert(iscellstr(s)&&ischar(pat) || ischar(s)&&iscellstr(pat),...
  'Invalid input arguments.');

caseinsens = myparse(varargin,'caseinsens',false);
if caseinsens
  fcn = @regexpi;
else
  fcn = @regexp;
end

if iscellstr(s)
  tf = cellfun(@(x)~isempty(fcn(x,pat,'once')),s);
else
  tf = cellfun(@(x)~isempty(fcn(s,x,'once')),pat);
end
