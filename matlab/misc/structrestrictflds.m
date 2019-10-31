function s0r = structrestrictflds(s0,s1)
% s0r = structrestrictflds(s0,s1)
% Restrict the fields of s0 to those present in s1
%
% s1 can also be a cellstr of fields

if isstruct(s1)
  s1 = fieldnames(s1);
end
s0r = rmfield(s0,setdiff(fieldnames(s0),s1));
