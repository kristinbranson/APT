function m = getenvall()
% Return a containers.Map of all environment variables and their values.
%
% Based on
% https://stackoverflow.com/questions/20004955/list-all-environment-variables-in-matlab

if ispc()
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

% Windows environment variables are case-insensitive
if ispc()
  keys = upper(keys);
end

% sort alphabetically
[keys,ord] = sort(keys);
vals = vals(ord);

m = containers.Map(keys,vals);

end  % function
