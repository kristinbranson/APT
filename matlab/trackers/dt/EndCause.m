classdef EndCause 
  % Enumeration type to support signaling what exactly brought training/tracking
  % to an end.
  enumeration
    complete  % training/tracking succeeded, completed normally
    abort     % training/tracking was abort by the user
    error     % training/tracking encountered an error and could not proceed
  end
end
