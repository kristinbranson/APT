function m = getenvall(method)

% Based on
% https://stackoverflow.com/questions/20004955/list-all-environment-variables-in-matlab

if nargin < 1, method = 'system'; end
method = validatestring(method, {'java', 'system'});

switch method
  case 'java'
    map = java.lang.System.getenv();  % returns a Java map
    keys = cell(map.keySet.toArray());
    vals = cell(map.values.toArray());
  case 'system'
    if ispc()
      %cmd = 'set "';  %HACK for hidden variables
      cmd = 'set';
    else
      cmd = 'env';
    end
    [~,out] = system(cmd);
    vars = regexp(strtrim(out), '^(.*)=(.*)$', ...
      'tokens', 'lineanchors', 'dotexceptnewline');
    vars = vertcat(vars{:});
    keys = vars(:,1);
    vals = vars(:,2);
end

% Windows environment variables are case-insensitive
if ispc()
  keys = upper(keys);
end

% sort alphabetically
[keys,ord] = sort(keys);
vals = vals(ord);

m = containers.Map(keys,vals);
