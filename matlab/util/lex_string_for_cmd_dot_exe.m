function result = lex_string_for_cmd_dot_exe(str)
  % Designed to mimic the lexing of strings by cmd.exe, in the context of a
  % wsl --exec bash -c <commmand-string> call.  That is, if str is the
  % <command-string>, this function outputs the command string as it will be
  % received by bash.
  %
  % These seem to be the rules:
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

  % Set the initial state for processing str.
  % See documentation of lex1(), below, for what these fields mean.
  state0 = struct('i', {0} , ...
                  'is_done', {false}, ...
                  'bs_count',{0}) ;  
  % Run str through a state machine that will consume one input character at a
  % time and output the result string.
  preresult = crank(@lex1, str, state0) ;
  if isempty(preresult) ,
    result = '' ;  % this is 0x0 instead of 1x0, which e.g. strcmp() cares about.
  else
    result = preresult ;
  end
end



function [yi, statei] = lex1(xi, statelast)
  % Evolution function to be used with crank() to parse strings like cmd.exe in
  % wsl --exec bash -c <command-string>.  In what follows, "dq" means the
  % doublequote character, and "bs" means the backslash character.
  
  % Unpack the state at the last time step
  i_last = statelast.i ;  % index of the previous character in the string
  is_done_last = statelast.is_done ;  % true iff we have already consumed the dq matching the initial dq
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
    % The string must end with a dq.
    % So if we have already consumed the final dq, and then there are more
    % characters, throw an error.    
    if is_done_last
      error('There are characters after the doublequote that matches the initial doublequote') ;
    end

    if was_within_bs_block ,
      % A backslash block has just concluded with a dq, so process it.
      if iseven(bs_count_last) ,
        % bs_count is even
        yi = repmat(bs, [1 bs_count_last/2]) ;  % curent dq marks end of string, so don't copy to output        
        is_done = true ;
        bs_count = 0 ;
          % the current dq marks the end of the string, so is_done==true
      else
        % bs_count is odd
        yi_bs = repmat(bs, [1 (bs_count_last-1)/2]) ;
        yi = horzcat(yi_bs, dq) ;  % the current dq was escaped, so add it to the output string
        is_done = false ;
        bs_count = 0 ;
      end
    else
      % We are not within a bs block
      % This dq marks the end of the string, unless it's at the start of the string.
      yi = '' ;  % don't copy either the first or last dq to output
      is_done = (i>1) ;
      bs_count = 0 ;
    end    
  elseif xi == bs
    % The string must end with a dq.
    % So if we have already consumed the final dq, and then there are more
    % characters, throw an error.    
    if is_done_last
      error('There are characters after the doublequote that matches the initial doublequote') ;
    end

    % The string has to start with a dq
    if i_last==0 ,
      error('First character must be a doublequote') ;
    end
    if was_within_bs_block ,
      % The bs block continues...
      yi = '' ;
      is_done = false ;
      bs_count = bs_count_last+1 ;
    else
      % Was not in a bs block before, therefore
      % a new bs block has started.
      yi = '' ;
      is_done = false ;
      bs_count = 1 ;
    end    
  elseif xi == eol ,
    % xi is the end-of-line character
    if ~is_done_last ,
      error('Last character must be an unescaped doublequote') ;
    end
    yi = '' ;
    is_done = true ;
    bs_count = 0 ;
  else
    % The string must end with a dq.
    % So if we have already consumed the final dq, and then there are more
    % characters, throw an error.    
    if is_done_last
      error('There are characters after the doublequote that matches the initial doublequote') ;
    end

    % xi is neither dq nor bs, nor EOL.
    % The string has to start with a dq.
    if i_last==0 ,
      error('First character must be a doublequote') ;
    end
    if was_within_bs_block ,
      % A bs block has just concluded with a non-dq, so process it.
      yi_bs = repmat(bs, [1 bs_count_last]) ;
      yi = horzcat(yi_bs, xi) ;
      is_done = false ;
      bs_count = 0 ;
    else
      % Characters besides bs and dq are generally passed as-is.
      yi = xi ;
      is_done = false ;
      bs_count = 0 ;
    end
  end

  % Pack up the state
  statei = struct('i', {i} , ...
                  'is_done', {is_done}, ...
                  'bs_count',{bs_count}) ;
end
