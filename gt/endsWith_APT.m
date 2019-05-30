function res = endsWith(s,suffix)

res = false;

nsuffix = numel(suffix);
if numel(s) < nsuffix,
  return;
end

res = strcmp(s(end-nsuffix+1:end),suffix);