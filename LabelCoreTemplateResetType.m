classdef LabelCoreTemplateResetType < uint32
  enumeration
    NORESET(0)
    RESET(10) % Reset with no tracking prediction ("white" template prediction)
    RESETPREDICTED(20) % Reset to tracking prediction
  end
end
