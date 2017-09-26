function [tf,loc] = tblismember(tblA,tblB,flds)
[tf,loc] = ismember(tblA(:,flds),tblB(:,flds));