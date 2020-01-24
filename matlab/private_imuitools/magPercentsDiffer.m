function differ = magPercentsDiffer(mag,candidateMag)
%magPercentsDiffer True if magnification factors lead to different percents.
%    DIFFER = magPercentsDiffer(MAG, CANDIDATE_MAG)
%
%    This function is used when deciding if a new CANDIDATE_MAG is significantly
%    different from the current mag MAG. Clients of this function depend on the
%    following conceptual rule:
%
%       Only take action when a CANDIDATE_MAG would lead to a different percent
%       and therefore different percent string being displayed in a
%       magnification box relative to that displayed for the current mag MAG.
%
%       Users believe that the magnification displayed in the magnification box
%       is the true magnification percent. Users expect to see changes in the
%       image magnification and in the magnification percent in tandem when they
%       do something to change the magnification.
%
%    Explanation is simplest by example.
%
%    Example 1
%    ---------
%
%    This function is used when zooming in/out to figure out the next
%    magnification so that zooming in/out will behave naturally for users.
%
%    For example, if a tool uses a magnification box which indicates the
%    magnification is 33%, this may correspond to any magnification factor that
%    would result in the string "33%" being displayed in the magnification box
%    such as: 0.33, 0.333333333333, 0.331, or even 0.325. If the magnification
%    box shows the sequence of magnifications as 33%, 50%, 67%, 100%,... then it
%    is natural for a user to expect a zoom-in-on-click gesture to take their
%    image from 33% to 50%.
%
%    If the magnification box shows "67%" that could correspond to 0.67,
%    0.6666666666, 0.671, or even 0.675. In all of these cases, a user would
%    expect the next zoom-in magnification to be 1.0 (100%) and the next
%    zoom-out magnification to be 50%.
%
%    And, similarly for zooming out.
%
%
%    Example 2
%    ---------
%
%    A user displays an image at 100% initially, and uses the zoom out tool to
%    go to 33%. They then type 33 <enter> in the magnification box and expect
%    that the image will not change magnification, even slightly.
%
%
%    Example 3
%    ---------
%
%    A user discovers that we impose a minimum magnification that depends on
%    image size. One way to see the minimum magnification value is to type 0
%    <enter> in the magnification box and after parsing, the minimum value will
%    be displayed. Users expect that repeated actions 0 <enter> will do the same
%    thing regardless of the current magnification.
%
%
%    See gecks 297981, 297740.

%   Copyright 2006-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2011/02/09 18:56:20 $

magPercent = javaMethodEDT('toPercent',...
    'com.mathworks.toolbox.images.MagnificationComboBox',...
    mag);
candidateMagPercent = javaMethodEDT('toPercent',...
    'com.mathworks.toolbox.images.MagnificationComboBox',...
    candidateMag);

% Return true any time the mag value and the candidateMag value would both lead
% to different magnification percents which would lead to a different string
% being displayed in the mag box.
differ = ~isequal(magPercent, candidateMagPercent);

