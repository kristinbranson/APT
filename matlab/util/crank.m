function [y, staten] = crank(f, x, state0)
  % Simulate a discrete-time state machine that begins with state0, and gets
  % input x(i) at time i.  The output yi at time i is determined by the function
  % f via [yi, state(i)] = f(x(i), state(i-1)).  yi is *concatenated* onto the
  % end of y so far, so yi need not be a scalar.  We append an end-of-line
  % character, char(0), to x before processing, so f should expect this and
  % handle it appropiately.  staten is the state output
  % from the last evaluation of f().  This function will be kinda slow, frankly.
  % x is assumed to be a row vector, and y will also be a row vector.  The state
  % can be almost anything as long as f() handles it correctly.

  % Add an end-of-line character to the end of the input
  eol = char(0) ;
  xz = horzcat(x, eol) ;

  % Prepare to process the input array one element at a time
  nin = numel(xz) ;
  if ischar(xz) ,
    y = repmat(' ', [1 nin]) ;  % preallocate, but will trim if needed at end
  else
    y = zeros([1 nin], class(xz)) ;  % preallocate, but will trim if needed at end
  end
  state = state0 ;
  nout = 0 ;

  % Process the input array one element at a time
  for i = 1 : nin ,
    state_last = state ;
    xi = xz(i) ;
    [yi, state] = feval(f, xi, state_last) ;
    nthis = numel(yi) ;
    y(nout+1:nout+nthis) = yi ;
    nout = nout + nthis ;
  end
  
  % Trim the output to the number of characters output
  y = y(1,1:nout) ;

  % Set the final state for return
  staten = state ;
end
