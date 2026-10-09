function test_yaml_inheritance()
  % yaml.ReadYaml resolves a 'parent:' reference that names a dotted path.
  %
  % The YAML library's doinheritance resolved each parent by building the
  % text 'oaroot.<path>' and eval'ing it.  That reads the nested field the
  % path names, which plain MATLAB can do with a dynamic field access per
  % path component; the eval form was the one construct in this library the
  % Python port's translator had to leave as a marker.  This test pins the
  % inheritance behaviour the rewrite must keep, on the library's own test
  % data: a child's own fields win over the parent's, later-listed parents
  % lose to earlier ones, a parent's parent is followed, and a parent that
  % does not exist raises MATLAB:MATYAML:NonExistentParent.

  dataDirPath = fullfile(APT.Root, 'matlab', '+yaml', 'Tests', 'Data', 'test_inheritance') ;

  s = yaml.ReadYaml(fullfile(dataDirPath, 'inheritance.yaml')) ;
  car1 = s.cars{1} ;  % parents: manufacturer.skoda, insurance.mandatory.Allianz, insurance.optional.Kooperativa
  assert(strcmp(car1.established, 'cca 1900'), 'car 1 did not inherit established from manufacturer.skoda') ;
  assert(strcmp(car1.place, 'MB'), 'car 1 did not inherit place from manufacturer.skoda') ;
  assert(strcmp(car1.ins_unit, 'CZK'), 'car 1 did not inherit ins_unit from insurance.mandatory.Allianz') ;
  assert(strcmp(car1.ins_note, 'IMPORTANT_1'), 'car 1''s own ins_note was overwritten by a parent''s') ;
  assert(isequal(car1.info.aaa, 2), 'car 1''s own info was overwritten by a parent''s') ;
  car2 = s.cars{2} ;  % parents: manufacturer.audi, insurance.mandatory.Allianz
  assert(strcmp(car2.place, 'Germany'), 'car 2 did not inherit place from manufacturer.audi') ;
  assert(isequal(car2.ins_price, 4000), 'car 2 did not inherit ins_price from insurance.mandatory.Allianz') ;

  s2 = yaml.ReadYaml(fullfile(dataDirPath, 'inheritance_multiple.yaml')) ;
  assert(isequal(s2.a.v1, 10), 'a''s own field was lost') ;
  assert(isequal(s2.a.w1, 111) && isequal(s2.a.w2, 798), 'a did not inherit from both b and c') ;
  assert(isequal(s2.a.x, 113), 'a did not inherit d''s field through b and c') ;

  % a parent path that does not exist
  missingParentFilePath = [tempname() '.yaml'] ;
  fid = fopen(missingParentFilePath, 'w') ;
  fprintf(fid, 'a:\n  v: 1\n  parent: nosuch.thing\nb:\n  w: 2\n') ;
  fclose(fid) ;
  cleaner = onCleanup(@()(delete(missingParentFilePath))) ;
  didError = false ;
  try
    yaml.ReadYaml(missingParentFilePath) ;
  catch ex
    didError = true ;
    assert(strcmp(ex.identifier, 'MATLAB:MATYAML:NonExistentParent'), ...
           'A missing parent raised %s rather than MATLAB:MATYAML:NonExistentParent', ex.identifier) ;
  end
  assert(didError, 'A missing parent did not raise') ;
end  % function
