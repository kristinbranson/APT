function c = struct2pvs(s)
if ~isstruct(s) || ~isscalar(s)
  error('Input must be a scalar struct.');
end

fn = fieldnames(s);
v = struct2cell(s);
c = [fn v]';
c = c(:);
