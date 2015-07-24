function TF = isRSetImage(hIm)
%isRSetImage returns true if hIm is an RSet image.
%
% TF = isRSetImage(hIm) returns TF, which is true if hIm is an R-Set.

%   Copyright 2008 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2008/11/24 14:58:43 $

TF = strcmp(get(hIm,'tag'),'rset overview');