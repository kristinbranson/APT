function show_units(h)

if isprop(h, 'Units') ,
  units = h.Units ;
  raw_tag = h.Tag ;
  tag = fif(isempty(raw_tag), '<no tag>', raw_tag) ;
  fprintf('tag: %s, units: %s\n', tag, units) ;
end

end