classdef EndCause 
  % Enumeration type to support signaling what exactly brought training/tracking
  % to an end.
  enumeration
    complete   % training/tracking succeeded, completed normally
    abort      % training/tracking was abort by the user
    error      % training/tracking encountered an error and could not proceed
    load       % loaded a tracker
    undefined  % training/tracking has not been run yet
  end
end
