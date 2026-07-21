function test_wait_bar_with_cancel()
% Test that WaitBarWithCancel works through a full lifecycle of periods.
% Guards against reliance on undocumented waitbar internals that changed in
% R2025a, where the graphical bar is no longer an hgjavacomponent.

wbObj = WaitBarWithCancel('Test title') ;
cleanupObj = onCleanup(@()(delete(wbObj))) ;  %#ok<NASGU>

% The handle to the graphical bar should be a valid scalar graphics object
assert(isscalar(wbObj.hBar) && isvalid(wbObj.hBar), ...
       'hBar is not a valid scalar handle') ;

% A period showing the completed fraction as numerator/denominator
wbObj.startPeriod('Doing stuff', 'shownumden', true, 'denominator', 10) ;
isCancelRequested = wbObj.updateFracWithNumDen(3) ;
assert(~isCancelRequested, 'Cancel should not have been requested') ;
assert(strcmp(wbObj.hTxt.String, 'Doing stuff (3/10)'), ...
       'Waitbar message is wrong: %s', wbObj.hTxt.String) ;
wbObj.endPeriod() ;

% A plain fractional period
wbObj.startPeriod('More stuff') ;
isCancelRequested = wbObj.updateFrac(0.5) ;
assert(~isCancelRequested, 'Cancel should not have been requested') ;
wbObj.endPeriod() ;

% During a nobar period the graphical bar should be hidden, and it should
% reappear when the enclosing period (which wants a bar) resumes
wbObj.startPeriod('Outer stuff') ;
wbObj.startPeriod('Quiet stuff', 'nobar', true) ;
assert(strcmp(char(get(wbObj.hBar, 'Visible')), 'off'), ...
       'Bar should be hidden during a nobar period') ;
wbObj.endPeriod() ;
assert(strcmp(char(get(wbObj.hBar, 'Visible')), 'on'), ...
       'Bar should be shown again after the nobar period ends') ;
wbObj.endPeriod() ;

% Cancel-disabled construction should also work
wbObj2 = WaitBarWithCancel('Test title 2', 'cancelDisabled', true) ;
cleanupObj2 = onCleanup(@()(delete(wbObj2))) ;  %#ok<NASGU>
wbObj2.startPeriod('Even more stuff') ;
wbObj2.updateFrac(0.25) ;
wbObj2.endPeriod() ;

fprintf('test_wait_bar_with_cancel passed.\n') ;

end  % function
