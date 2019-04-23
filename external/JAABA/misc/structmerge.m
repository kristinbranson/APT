function s = structmerge(varargin)
% s = structmerge(s1,s2,s3,...)
% Merge structures into one combined structure.

s = struct();
for c = 1:numel(varargin)
    stmp = varargin{c};
    assert(isscalar(stmp) && isstruct(stmp),...
        'All input arguments must be scalar structures.');
    tmpflds = fieldnames(stmp);
    for fld = tmpflds(:)'
        fld = fld{1}; %#ok<FXSET>
        if isfield(s,fld)
            warning('structmerge:overlappingFields',...
                'Overwriting field ''%s''.',fld);
        end
        s.(fld) = stmp.(fld);
    end
end
    