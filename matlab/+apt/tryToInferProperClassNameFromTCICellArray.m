function fixedClassName = tryToInferProperClassNameFromTCICellArray(ca)
% Called when loading an MA project with a TCI cell array that has ca{1} of
% 'DeepTracker'.
% In theory this should not happen, but there was once a buggy version of APT
% that sometimes used this className for MA projects.
% The result is the inferred class name, either 'DeepTrackerBottomUp' or 
% 'DeepTrackerTopDown'.
% Throws an error if it can't infer a class name with some degree of
% confidence.

caLength = numel(ca) ;
if caLength < 3
  error('A cell array that specifies a tracker type seems to be malformed.  If should have at least three elements, but has %d elements', ...
        caLength) ;
end
element3 = ca{3} ;  
  % This is the critical element.  If it's a DLNetType, this should be a BU
  % tracker.  If it's a cell array, it should be a TD tracker.
if iscell(element3)
  if numel(ca{2})>=4 && numel(ca{3})>=4
    fixedClassName = 'DeepTrackerTopDown' ;
  else
    error(['A cell array that specifies a tracker type seems to be malformed.  ' ...
           'It looks like it should be a top-down tracker, but at least one of the elements is too short.'], caLength) ;              
  end
elseif isa(element3, 'DLNetType')
  if caLength < 5
    error(['A cell array that specifies a tracker type seems to be malformed.  ' ...
           'It has %d elements, when given the type element3 it should have at least five elements.'], caLength) ;
  end
  netMode = ca{5} ;
  if netMode ~= DLNetMode.multiAnimalBU
    error(['A cell array that specifies a tracker type seems to be malformed.  ' ...
           'As far as I can tell, its fifth element should be DLNetMode.multiAnimalBU but is not.']) ;
  end
  % If we get here we are good.
  fixedClassName = 'DeepTrackerBottomUp' ;
else
  error(['A cell array that specifies a tracker type seems to be malformed.  ' ...
         'The third element should be a cell array or of type DLNetType, but it is neither.']) ;
end

end  % function
