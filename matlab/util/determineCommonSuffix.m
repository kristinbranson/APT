function [oldPrefix, newPrefix, commonSuffix] = determineCommonSuffix(oldPath, newPath)
% Determine the common suffix in the two paths oldPath and newPath.  Paths are
% first standardized to use the current platform's file separator, then the
% longest common string at the end of the two paths is found.  This is
% returned in commonSuffix, and the (differing) prefixes of the two paths are
% returned in oldPrefix and newPrefix, respectively.

oldMovFileFullStandardized = standardizeFileSeparators(oldPath);
newMovFileFullStandardized = standardizeFileSeparators(newPath);
minLength = min(numel(oldMovFileFullStandardized),numel(newMovFileFullStandardized));
matchingPath = oldMovFileFullStandardized(end-minLength+1:end) == newMovFileFullStandardized(end-minLength+1:end);
eIndex = find(matchingPath==0,1,'last');
oldPrefix = oldMovFileFullStandardized(1:end-minLength+eIndex);
newPrefix = newMovFileFullStandardized(1:end-minLength+eIndex);
commonSuffix = oldMovFileFullStandardized(end-minLength+eIndex+1:end);

end  % function
