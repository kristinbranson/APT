function result = escape_string_for_cmd_dot_exe(str)
  % Process the string str so that when the result is passed as part of a
  % Windows cmd.exe command line, in the context of a wsl --exec bash -c
  % <commmand-string> call, it will be interpreted as a single token, and the
  % string received by bash will be identical to str
  %
  % These seem to be the rules for how cmd.exe parses such strings:
  %
  % * A double quote mark preceded by a backslash (\") is interpreted as a
  %   literal double quote mark (").
  % 
  % * Backslashes are interpreted literally, unless they immediately precede a
  %   double quote mark.
  % 
  % * If an even number of backslashes is followed by a double quote mark, then
  %   one backslash (\) is placed in the argv array for every pair of backslashes
  %   (\\), and the double quote mark (") is interpreted as a string delimiter.
  % 
  % * If an odd number of backslashes is followed by a double quote mark, then
  %   one backslash (\) is placed in the argv array for every pair of backslashes
  %   (\\). The double quote mark is interpreted as an escape sequence by the
  %   remaining backslash, causing a literal double quote mark (") to be placed in
  %   argv.
  %  
  % Those rules are taken from:
  %   https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args?view=msvc-170&redirectedfrom=MSDN#parsing-c-command-line-arguments
  %
  % This function should be the inverse of parse_string_for_cmd_dot_exe().
  % That is,
  %
  %   isequal(str,
  %           parse_string_for_cmd_dot_exe(escape_string_for_cmd_dot_exe(str)))
  %
  % should hold for all old-school strings str.

  % Set the initial state for processing str.
  % See documentation of escape1(), below, for what these fields mean.
  state0 = struct('i', {0} , ...
                  'bs_count',{0}) ;  
  % Run str through a state machine that will consume one input character at a
  % time and output the result string.
  preresult = crank(@escape1, str, state0) ;
  result = sprintf('"%s"', preresult) ;
end



function [yi, statei] = escape1(xi, statelast)
  % Evolution function to be used with crank() to parse escape string for use
  % with  cmd.exe in wsl --exec bash -c <command-string>.  In what follows, "dq"
  % means the doublequote character, and "bs" means the backslash character.
  
  % Unpack the state at the last time step
  i_last = statelast.i ;  % index of the previous character in the string
  bs_count_last = statelast.bs_count ;
    % the number of bs's consumed in the current bs block so far
    % If not in a bs block, equal to zero.

  dq = '"' ;
  bs = '\' ;
  eol = char(0) ;
  i = i_last+1 ;
  was_within_bs_block = (bs_count_last>0)  ;
    % true iff we are currently within a block of bs characters.
  if xi == dq
    % Escape the dq, taking into account however many bs's preceded it.
    yi_bs = repmat(bs, [1 2*bs_count_last+1]) ;        
    yi = horzcat(yi_bs, dq) ;
    bs_count = 0 ;
  elseif xi == bs
    if was_within_bs_block ,
      % The bs block continues...
      yi = '' ;
      bs_count = bs_count_last+1 ;
    else
      % Was not in a bs block before, therefore
                                                                          % a new bs block has started.
      yi = '' ;
      bs_count = 1 ;
    end    
  elseif xi == eol
    if was_within_bs_block ,
      % A bs block has just concluded with eol, so it will have a dq after it, so we
      % need to double it.
      yi = repmat(bs, [1 2*bs_count_last]) ;
      bs_count = 0 ;
    else
      % Don't emit any characters for eol outside a bs block
      yi = '' ;
      bs_count = 0 ;
    end
  else
    % xi is neither dq nor bs.
    if was_within_bs_block ,
      % A bs block has just concluded with a non-dq, so the bs block can be
      % reproduced as-is.
      yi_bs = repmat(bs, [1 bs_count_last]) ;
      yi = horzcat(yi_bs, xi) ;
      bs_count = 0 ;
    else
      % Characters besides bs and dq are generally passed as-is.
      yi = xi ;
      bs_count = 0 ;
    end
  end

  % Pack up the state
  statei = struct('i', {i} , ...
                  'bs_count',{bs_count}) ;
end
