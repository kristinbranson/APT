function r = struct2rowsexport(s,varargin)
% r = struct2rowsexport(s,varargin)
% Optional PVs:
%   - nansAsStrings: If true, numeric scalar NaNs are replaced with the
%   string 'NaN'. Defaults to false.

nansAsStrings = myparse(varargin,...
  'nansAsStrings',false);

assert(isstruct(s)&&isvector(s),'Expected d to be a struct vector.');
s = s(:);

r = struct2cell(s);
r = r';

% replace values in r for export
for c = 1:numel(r)
  val = r{c};
  if isscalar(val)&&isnan(val)&&nansAsStrings
    r{c} = 'NaN';
  elseif isnumeric(val)&&isscalar(val) || ...
      islogical(val)&&isscalar(val) || ...
      ischar(val)&&isrow(val) || ...
      isequal(val,[])
    % none
  else
    r{c} = '<array value>';
  end
end

hdr = fieldnames(s);
r = [hdr(:)';r];
