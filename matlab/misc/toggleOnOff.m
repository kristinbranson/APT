function toggleOnOff(h,prop)
switch h.(prop)
  case 'on'
    h.(prop) = 'off';
  case 'off'
    h.(prop) = 'on';
  otherwise
    assert(false);
end