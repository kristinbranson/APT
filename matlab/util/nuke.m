% Delete every consarned thing.
% There are probably more kinds of user-managed object classes (as contrasted with
% scoped object classes) that should be found and deleted here, but these are
% the only kinds we use.  According to
%
%   https://blogs.mathworks.com/loren/2008/07/29/understanding-object-cleanup/
%
% all such classes have a <class name>find() and/or <class name>findall()
% function that can be used to find all such objects and delete them.

delete(timerfindall()) ;
delete(findall(groot(), 'Type', 'figure')) ;
clear classes  %#ok<CLCLS>
clear java  %#ok<CLJAVA>
