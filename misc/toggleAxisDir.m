function d = toggleAxisDir(d)
switch d
  case 'normal'
    d = 'reverse';
  case 'reverse'
    d = 'normal';
  otherwise
    assert(false);
end
end
   