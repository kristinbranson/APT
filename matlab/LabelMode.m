classdef LabelMode
  properties 
    prettyString
  end
  enumeration
    NONE ('None')
    SEQUENTIAL ('Sequential')
    TEMPLATE ('Template')
    HIGHTHROUGHPUT ('HighThroughput')
%     ERRORCORRECT ('ErrorCorrect')
    MULTIVIEWCALIBRATED2 ('Multiview')
    MULTIANIMAL ('MultiAnimal')
    SEQUENTIALADD ('SequentialAdd')
  end
  methods
    function obj = LabelMode(pStr)
      obj.prettyString = pStr;
    end
    function tf = multiviewOnly(obj)
      switch obj
        case {LabelMode.MULTIVIEWCALIBRATED2}
          tf = true;
        otherwise
          tf = false;
      end
    end
  end
end