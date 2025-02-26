function pauseTight(delayInterval)
%PAUSETIGHT A command that completely blocks Matlab execution access for specified delay. Unlike the built-in pause() command, callbacks are also prevented from firing.
%
%% NOTES
%   The pauseTight() function is primarily intended as a fully blocking delay, in contrast to the built-in pause() command, which allows callbacks to fire.
%   This function can be useful when polling hardware -- i.e. as a means for airtight polling for a hardware event without talking to the hardware at a very high rate
%
%% CHANGES
%   VI041111A: Handle case of empty delayInterval value as specifying no delay (matches native pause() behavior)
%
%% CREDITS
%   Created 3/30/10, by Vijay Iyer
%% ******************************************************

h1 = tic();

%%%VI041111A
if isempty(delayInterval)
    return;
end

while true
    if toc(h1) > delayInterval
        break;
    end
end

end

