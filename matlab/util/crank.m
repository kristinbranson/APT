function [y, staten] = crank(f, x, state0)
  % Simulate a discrete-time state machine that begins with state0, and gets
  % input x(i) at time i.  The output yi at time i is determined by the function
  % f via [yi, state(i)] = f(x(i), state(i-1)).  yi is *concatenated* onto the
  % end of y so far, so yi need not be a scalar.  staten is the state output
  % from the last evaluation of f().  This function will be kinda slow, frankly.
  % x is assumed to be a row vector, and y will also be a row vector.  The state
  % can be almost anything as long as f() handles it correctly.

  nin = numel(x) ;
  state = state0 ;
  if ischar(x) ,
    y = repmat(' ', [1 nin]) ;  % preallocate, but will trim if needed at end
  else
    y = zeros([1 nin], class(x)) ;  % preallocate, but will trim if needed at end
  end
  nout = 0 ;
  for i = 1 : nin ,
    state_last = state ;
    xi = x(i) ;
    [yi, state] = feval(f, xi, state_last) ;
    nthis = numel(yi) ;
    y(nout+1:nout+nthis) = yi ;
    nout = nout + nthis ;
  end
  y = y(1,1:nout) ;  % trim if too long
  staten = state ;
end
