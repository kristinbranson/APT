function [tf,loc] = tblismember(tblA,tblB,flds)
% Finds rows of tblA that match some row of tblB in all fields (aka columns)
% named in flds.
%
% tf is a logical col vector with one element for each row of tblA, 
%    which is true iff that row of tblA matches some row of tblB
% loc is a numeric col vector with one element for each row of tblA.
%    For elements where tf is true, tf(i) this gives the row index into tblB that
%    matches row i of tblA.

% Added these to deal with occasional type mismatches with these columns
% Apparently ismember() is happy to compare anything to a double.
if isTableColumn(tblA, 'frm')
  tblA.frm = double(tblA.frm) ;
end
if isTableColumn(tblA, 'iTgt')
  tblA.iTgt = double(tblA.iTgt) ;
end

vA = tblA(:,flds);
vB = tblB(:,flds);

for i = 1:numel(flds),
  dataA = vA(:,i).Variables;
  coliscell = iscell(dataA) && size(dataA,2) > 1 && all(all(cellfun(@ischar,dataA)));
  if coliscell,
    newdata = dataA(:,1);
    for j = 2:size(dataA,2),
      newdata = cellfun(@(x0,x1) [x0,'__',x1],newdata,dataA(:,j),'Uni',0);
    end
    vA.(flds{i}) = newdata;
  end
  dataB = vB(:,i).Variables;
  coliscell = iscell(dataB) && size(dataB,2) > 1 && all(all(cellfun(@ischar,dataB)));
  if coliscell,
    newdata = dataB(:,1);
    for j = 2:size(dataB,2),
      newdata = cellfun(@(x0,x1) [x0,'__',x1],newdata,dataB(:,j),'Uni',0);
    end
    vB.(flds{i}) = newdata;
  end
end
[tf,loc] = ismember(vA,vB);

end
