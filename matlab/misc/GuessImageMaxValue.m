function result = GuessImageMaxValue(im)
% Guess a good maximum value for the display of image im, which is typically
% the first frame of a movie.
% This version improves on the old version by first branching on the class of im.
% In particular GuessImageMaxValue(uint8(0)) => uint(255), not uint8(0).

if isa(im, 'uint8')
  result = uint8(255) ;
elseif isa(im, 'uint16')
  maxv = max(im(:));
  if maxv == 0
    result = uint16(65535) ;
  elseif maxv < 32768 
    result = maxv ;
  else
    result = uint16(65535) ;
  end    
elseif isa(im, 'double') || isa(im, 'single')
  maxv = max(im(:));
  if maxv == 0
    result = cast(1, class(im)) ;
  elseif maxv < 0.5 
    result = maxv ;
  elseif maxv <= 1
    result = cast(1, class(im)) ;
  elseif maxv <= 255
    result = cast(255, class(im)) ;
  elseif maxv <= 65535
    if maxv < 32768
      result = maxv ;
    else
      result = cast(65535, class(im)) ;
    end        
  else
    % maxv > 65535 
    result = maxv ;
  end    
else
  % What the what
  warningNoTrace('Type of im is %s, which is odd.  Converting to double', class(im)) ;
  result = GuessImageMaxValue(double(im)) ;
end
