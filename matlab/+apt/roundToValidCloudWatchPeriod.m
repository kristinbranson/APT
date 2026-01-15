function result = roundToValidCloudWatchPeriod(periodsec)
% Round periodsec to a valid CloudWatch period: 10, 20, 30, or a multiple of 60.
% Recent versions of AWS CloudWatch require the period to be one of these values.

if periodsec <= 30
  validValues = [10 20 30] ;
  [~, idx] = min(abs(validValues - periodsec)) ;
  result = validValues(idx) ;
elseif periodsec < 60
  % Between 30 and 60: pick whichever is closer
  if periodsec - 30 <= 60 - periodsec
    result = 30 ;
  else
    result = 60 ;
  end
else
  result = round(periodsec / 60) * 60 ;
  if result == 0
    result = 60 ;
  end
end

end
