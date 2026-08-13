function nlbls = readNLabels(tpjson)
  % Read the per-movie label counts from a training-package json file.
  tp = TrnPack.hlpLoadJson(tpjson);
  nlbls = arrayfun(@(x)size(x.pabs,2),tp.locdata);
end % function
