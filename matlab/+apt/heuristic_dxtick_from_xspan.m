function dxtick = heuristic_dxtick_from_xspan(raw_xspan)
  % Want dxtick to be close to xspan/10 and to be a positive real, and to be a
  % power of ten, or be two times a power of ten, or be five times a power of ten.
  % This guarantees the number of ticks per span will in the interval (6,18).
  assert(isnumeric(raw_xspan) && isscalar(raw_xspan) && raw_xspan>0, 'Illegal argument') ;
  xspan = double(raw_xspan) ;
  ideal_dxtick = xspan/10 ;
  ideal_p = log10(ideal_dxtick) ;  % ideal power of ten, not generally an integer
  low_dxtick_option = 10^floor(ideal_p) ;
  dxtick_from_option_index = low_dxtick_option * [1 2 5 10] ;
  absdiff_from_option_index = abs(dxtick_from_option_index-ideal_dxtick) ;
  [~,option_index] = min(absdiff_from_option_index) ;
  dxtick = dxtick_from_option_index(option_index) ;
end
