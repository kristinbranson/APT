function test_ufmf_write_struct_error_message()
  % ufmf_write_struct() must refuse a field it cannot encode with an error
  % that names the value's class.
  %
  % Regression test: the error path called classname(value), which is not a
  % MATLAB function, so the caller saw "Unrecognized function or variable
  % 'classname'" instead of the intended message.

  fileName = [tempname() '.ufmf'] ;
  fid = fopen(fileName, 'w') ;
  assert(fid > 0, 'Could not open a temporary file') ;
  cleaner = onCleanup(@()(cleanUp(fid, fileName))) ;

  index = struct('frame', struct('loc', 1), 'note', 'not encodable') ;
  didThrow = false ;
  try
    ufmf_write_struct(fid, index) ;
  catch me
    didThrow = true ;
    assert(contains(me.message, 'Unable to write entity of class char'), ...
           'Wrong error for a char field: %s', me.message) ;
  end
  assert(didThrow, 'A char field was written without error') ;
end  % function

function cleanUp(fid, fileName)
  % Close the temporary file and remove it.
  fclose(fid) ;
  if exist(fileName, 'file')
    delete(fileName) ;
  end
end  % function
