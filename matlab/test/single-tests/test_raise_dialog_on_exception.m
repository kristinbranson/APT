function test_raise_dialog_on_exception()
  % raiseDialogOnException() shows a dialog for an exception: titled
  % 'Warning' when the identifier starts with 'APT:warningsOccurred' and
  % there is at most one cause, 'Warnings' for more causes, and 'Error'
  % for any other exception.
  %
  % Regression test: the warning title was chosen with ws.fif(), a
  % wavesurfer function APT does not ship, so the warning path errored
  % instead of showing the dialog.

  figuresBefore = findall(0, 'Type', 'figure') ;
  cleaner = onCleanup(@()(deleteNewFigures(figuresBefore))) ;

  singleWarning = MException('APT:warningsOccurred', 'One thing went wrong') ;
  raiseDialogOnException(singleWarning) ;
  dialog = newFigureNamed(figuresBefore, 'Warning') ;
  assert(isscalar(dialog), 'No ''Warning'' dialog appeared for a single warning.') ;
  delete(dialog) ;

  twoWarnings = MException('APT:warningsOccurred', 'Two things went wrong') ;
  twoWarnings = twoWarnings.addCause(MException('APT:firstThing', 'First thing')) ;
  twoWarnings = twoWarnings.addCause(MException('APT:secondThing', 'Second thing')) ;
  raiseDialogOnException(twoWarnings) ;
  dialog = newFigureNamed(figuresBefore, 'Warnings') ;
  assert(isscalar(dialog), 'No ''Warnings'' dialog appeared for two warnings.') ;
  delete(dialog) ;

  plainError = MException('APT:somethingElse', 'Not a warning') ;
  raiseDialogOnException(plainError) ;
  dialog = newFigureNamed(figuresBefore, 'Error') ;
  assert(isscalar(dialog), 'No ''Error'' dialog appeared for an ordinary exception.') ;
  delete(dialog) ;
end  % function

function result = newFigureNamed(figuresBefore, name)
  % The figures with the given Name that did not exist before the test.
  figuresNow = findall(0, 'Type', 'figure', 'Name', name) ;
  if isempty(figuresBefore) ,
    result = figuresNow ;
  else
    result = figuresNow(~ismember(figuresNow, figuresBefore)) ;
  end
end  % function

function deleteNewFigures(figuresBefore)
  % Close whatever the test left open.
  figuresNow = findall(0, 'Type', 'figure') ;
  if isempty(figuresBefore) ,
    delete(figuresNow) ;
  else
    delete(figuresNow(~ismember(figuresNow, figuresBefore))) ;
  end
end  % function
