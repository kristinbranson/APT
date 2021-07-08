function lims = fixLim(lims)
if isnan(lims(1))
  lims(1) = -inf;
end
if isnan(lims(2))
  lims(2) = inf;
end