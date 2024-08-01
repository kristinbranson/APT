function result = merge_struct(p, s, donotmerge, deep)
import yaml.*;
if ~( isstruct(p) && isstruct(s) )
        error('Only structures can be merged.');
    end;
    if ~exist('donotmerge','var')
        donotmerge = {};
    end
    if ~exist('deep','var')
        deep = 0;
    elseif strcmp(deep, 'deep')
        deep = 1;
    end;
    result = p;
    for i = fields(s)'
        fld = char(i);
        if any(cellfun(@(x)isequal(x, fld), donotmerge))
            continue;
        end;
        if deep == 1 && isfield(result, fld) && isstruct(result.(fld)) && isstruct(s.(fld))
            result.(fld) = merge_struct(result.(fld), s.(fld), donotmerge, deep);
        else
            result.(fld) = s.(fld);
        end;
    end;
end
