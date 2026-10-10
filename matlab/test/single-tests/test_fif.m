function test_fif()
  % fif(test, trueValue, falseValue) is the functional if: a scalar test
  % picks one whole value, and an array test picks element by element,
  % promoting a scalar branch to the test's size.  The array form is the
  % contract of wavesurfer's ws.fif(), which raiseDialogOnException() was
  % written against; APT's fif() has it and the promotion besides.

  % A scalar test picks a whole value, of any class and size
  assert(strcmp(fif(true, 'yes', 'no'), 'yes')) ;
  assert(strcmp(fif(false, 'yes', 'no'), 'no')) ;
  assert(strcmp(fif(1 <= 1, 'Warning', 'Warnings'), 'Warning')) ;
  assert(strcmp(fif(2 <= 1, 'Warning', 'Warnings'), 'Warnings')) ;
  assert(isequal(fif(true, [1 2 3], []), [1 2 3])) ;
  assert(isequal(fif(false, [1 2 3], []), [])) ;

  % An array test picks element by element from same-sized branches
  assert(isequal(fif([true false true], [1 2 3], [10 20 30]), [1 20 3])) ;
  assert(isequal(fif(logical([0 1 ; 1 0]), [1 2 ; 3 4], [5 6 ; 7 8]), [5 2 ; 3 8])) ;

  % ...promoting a scalar branch to the test's size
  assert(isequal(fif([true false], 7, [5 6]), [7 6])) ;
  assert(isequal(fif(logical([0 1 ; 1 0]), [1 2 ; 3 4], 0), [0 2 ; 3 0])) ;
end  % function
