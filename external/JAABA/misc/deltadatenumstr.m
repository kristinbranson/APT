function str = deltadatenumstr(dt)
% str = deltadatenumstr(dt)
% Nice string representing time interval. Input is in days.

ONEHOUR = 1/24; % days
ONEMINUTE = ONEHOUR/60;
ONESECOND = ONEMINUTE/60;

if dt > 1
  str = sprintf('%d days',round(dt));
elseif dt>ONEHOUR
  str = sprintf('%d hours',round(dt/ONEHOUR));
elseif dt > ONEMINUTE
  str = sprintf('%d minutes',round(dt/ONEMINUTE));  
elseif dt > ONESECOND
  str = sprintf('%d seconds',round(dt/ONESECOND));  
else
  str = 'less than 1 second';  
end