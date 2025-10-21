classdef DLTrainType
  % Enumeration to represent whether a training bout is a new training bout, or
  % a restart of a training bout (so that that database does not need to be recreated).

  % But at present I don't think we ever create a DLTrainType.Restart value. 
  % -- ALT, 2025-10-08
  enumeration 
    New
    Restart
    % RestartAug
  end
end
