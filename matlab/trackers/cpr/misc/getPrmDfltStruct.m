function prm = getPrmDfltStruct(args,defaults)
% Convenience wrapper for getPrmDflt
% args: cellstr
% defaults: struct
%
% prm: struct

assert(iscell(args));
assert(isstruct(defaults));
dflt = [fieldnames(defaults) struct2cell(defaults)]';
dflt = dflt(:);
prm = getPrmDflt(args,dflt(:)',true);
