function tfsucc = waitforPoll(pollCbk,iterwaittime,maxwaittime)
% Poll a callback until it returns true or timeout occurs
%
% pollCbk: fcn handle that returns a scalar logical
% iterwaittime, maxwaittime: in seconds

starttime = tic;

tfsucc = false;
while true
  tf = feval(pollCbk);
  if tf
    tfsucc = true;
    break;
  end
  if toc(starttime) > maxwaittime
    return;
  end
  pause(iterwaittime);
end
