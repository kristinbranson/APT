function tokens = bashLex(inputString)
  % Lex a Matlab string using bash shell tokenization rules
  % 
  % inputString: row char array to tokenize
  % tokens: cell array of strings, each element is one token
  
  if ischar(inputString) && (isempty(inputString) || isrow(inputString))
    % all is well
  else
    error('Input must be a row char array');
  end
  
  tokens = cell(1,0);
  currentToken = repmat(' ', [1 1024]);
  currentTokenLength = 0;
  tfIsInSingleQuotes = false;
  tfIsInDoubleQuotes = false;
  tfIsEscaped = false;
  
  i = 1;
  while i <= length(inputString)
    char = inputString(i);
    
    if tfIsEscaped
      currentTokenLength = currentTokenLength + 1;
      currentToken(currentTokenLength) = char;
      tfIsEscaped = false;
    elseif char == ''''
      if tfIsInDoubleQuotes
        currentTokenLength = currentTokenLength + 1;
        currentToken(currentTokenLength) = char;
      elseif tfIsInSingleQuotes
        tfIsInSingleQuotes = false;
      else
        tfIsInSingleQuotes = true;
      end
    elseif char == '"'
      if tfIsInSingleQuotes
        currentTokenLength = currentTokenLength + 1;
        currentToken(currentTokenLength) = char;
      elseif tfIsInDoubleQuotes
        tfIsInDoubleQuotes = false;
      else
        tfIsInDoubleQuotes = true;
      end
    elseif char == '\'
      if tfIsInSingleQuotes || tfIsInDoubleQuotes
        currentTokenLength = currentTokenLength + 1;
        currentToken(currentTokenLength) = char;
      else
        tfIsEscaped = true;
      end
    elseif isspace(char)
      if tfIsInSingleQuotes || tfIsInDoubleQuotes
        currentTokenLength = currentTokenLength + 1;
        currentToken(currentTokenLength) = char;
      else
        if currentTokenLength > 0
          tokens{1,end+1} = currentToken(1:currentTokenLength);  %#ok<AGROW>
          currentTokenLength = 0;
        end
      end
    else
      currentTokenLength = currentTokenLength + 1;
      currentToken(currentTokenLength) = char;
    end
    
    i = i + 1;
  end
  
  if currentTokenLength > 0
    tokens{1,end+1} = currentToken(1:currentTokenLength);
  end
  
  if tfIsInSingleQuotes || tfIsInDoubleQuotes
    error('Unterminated quote in input string');
  end
end