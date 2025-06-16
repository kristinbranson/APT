function test_windows_escaping()
  % Tests that the function escape_string_for_cmd_dot_exe().
  % escape_string_for_cmd_dot_exe() should be the inverse of parse_string_for_cmd_dot_exe().
  % That is,
  %
  %   isequal(str,
  %           parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
  %
  % should hold for all old-school strings str.

  % Seed the RNG for reproducibility.
  % Restore it afterwards.  Can't hurt...
  t = rng() ;
  rng(42) ;
  oc = onCleanup(@()(rng(t))) ;

  % Test some particular strings
  strs = { '', 'a', '\', '"', '\"', '"\', 'Hello, world!\', 'foo bar \\\\"\\', 'echo foo && false ; echo $?' } ;
  for i = 1 : numel(strs) ,
    str = strs{i} ;
    if ~strcmp(str, parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
      error('Output of parse(escape(str)) not equal to str for str == ''%s''', str) ;
    end
  end

  % Test short sequences of backslashes, doublequotes
  bs = '\' ;
  dq = '"' ;
  for bs_count = 0:5
    for dq_count = 0:5
      bss = repmat(bs, [1 bs_count]) ;
      dqs = repmat(dq, [1 bs_count]) ;
      str = norm_string_if_empty(horzcat(bss,dqs)) ;
      if ~strcmp(str, parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
        error('Output of parse(escape(str)) not equal to str for str == ''%s''', str) ;
      end
      str = norm_string_if_empty(horzcat(dqs,bss)) ;
      if ~strcmp(str, parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
        error('Output of parse(escape(str)) not equal to str for str == ''%s''', str) ;
      end
    end
  end

  % Test random sequences of bs, dq
  len = 10 ;
  n = 100 ;
  for i = 1 : n
    p = rand([1 len]) ;
    is_dq = (p<0.5) ;
    str = arrayfun(@(is_dq)(fif(is_dq, dq, bs)), is_dq) ;
    if ~strcmp(str, parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
      error('Output of parse(escape(str)) not equal to str for str == ''%s''', str) ;
    end
  end

  % Test random sequences of bs, dq, and another letter
  len = 20 ;
  n = 100 ;  
  for i = 1 : n
    q = rand([1 len]) ;
    str = arrayfun(@(p)(fif(p<1/3, dq, fif(p<2/3, bs, 'a'))), q) ;
    if ~strcmp(str, parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
      error('Output of parse(escape(str)) not equal to str for str == ''%s''', str) ;
    end
  end
end  % function
