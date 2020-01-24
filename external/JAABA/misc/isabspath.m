function v = isabspath(filename)

if isempty(filename),
  v = false;
  return;
end

if isunix,
  v = filename(1) == '/';
else
  v = numel(filename) > 1 && filename(2) == ':';
end