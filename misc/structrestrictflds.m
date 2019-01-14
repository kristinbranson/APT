function s0r = structrestrictflds(s0,s1)
% s0r = structrestrictflds(s0,s1)
% Restrict the fields of s0 to those present in s1
s0r = rmfield(s0,setdiff(fieldnames(s0),fieldnames(s1)));
