function set_pixel_units_to_normalized(h)

if isprop(h, 'Units') && strcmpi(h.Units, 'pixels') ,
  h.Units = 'normalized' ;
end

end
