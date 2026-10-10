function test_encode_for_persistence_struct()
  % encode_for_persistence() encodes a struct field by field, keeping its
  % size and field names, and decode_encoding_container() gets the struct
  % back.
  %
  % Regression test: the struct branch called ws.structWithDims() and
  % ws.encodeAnythingForPersistence(), wavesurfer functions APT does not
  % ship (APT's own are struct_with_dims() and encode_for_persistence()
  % itself), so encoding any struct errored.

  % A scalar struct holding the built-in types
  scalarStruct = struct('count', 3, 'name', 'fly', 'isGood', true, 'items', {{1, 'two'}}) ;
  checkRoundTrip(scalarStruct) ;

  % A struct array keeps its shape
  structArray = struct('x', {1, 2, 3 ; 4, 5, 6}) ;
  assert(isequal(size(structArray), [2 3])) ;
  checkRoundTrip(structArray) ;

  % Nested structs
  nestedStruct = struct('inner', struct('a', 1, 'b', 'bee'), 'outer', 2) ;
  checkRoundTrip(nestedStruct) ;

  % An empty struct array keeps its field names
  emptyWithFields = struct('p', {}, 'q', {}) ;
  checkRoundTrip(emptyWithFields) ;

  % A struct with no fields
  checkRoundTrip(struct()) ;

  % Unwrapped, a struct of built-in values encodes to itself
  plainStruct = struct('a', 1, 'b', 'bee') ;
  unwrapped = encode_for_persistence(plainStruct, false) ;
  assert(isequal(unwrapped, plainStruct), ...
         'The unwrapped encoding of a plain struct should equal the struct.') ;
end  % function

function checkRoundTrip(original)
  % Encode, check the container, decode, compare.
  encoded = encode_for_persistence(original) ;
  assert(is_an_encoding_container(encoded), ...
         'encode_for_persistence() did not return an encoding container.') ;
  assert(strcmp(encoded.class_name, 'struct'), ...
         'The container''s class_name should be ''struct'', was ''%s''.', encoded.class_name) ;
  assert(isequal(size(encoded.encoding), size(original)), ...
         'The encoding does not have the struct''s size.') ;
  assert(isequal(fieldnames(encoded.encoding), fieldnames(original)), ...
         'The encoding does not have the struct''s field names.') ;
  decoded = decode_encoding_container(encoded) ;
  assert(isequal(decoded, original), 'The struct did not survive a round trip.') ;
end  % function
