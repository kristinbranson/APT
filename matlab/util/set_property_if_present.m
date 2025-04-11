function set_property_if_present(h, name, value)

if isprop(h, name) ,
  h.(name) = value ;
end

end
