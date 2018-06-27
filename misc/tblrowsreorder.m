function [tbl,idx] = tblrowsreorder(tbl,tblRef,fld)
% Reorder rows of tbl to match those of tblRef wrt fld.
%
% tbl: table
% tblRef: reference table
% fld: single field present in both tbl and tblRef comparable via
%   ismember()
%
% tbl (out): table, with rows reordered
% idx: tblout = tbl(idx,:)

[~,idx] = ismember(tblRef.(fld),tbl.(fld));
tbl = tbl(idx,:);
assert(isequal(tblRef.(fld),tbl.(fld)));
