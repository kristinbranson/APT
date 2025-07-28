function [tf,loc] = tblismember(tblA,tblB,flds)


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
