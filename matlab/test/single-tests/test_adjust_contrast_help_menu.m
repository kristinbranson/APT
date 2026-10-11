function test_adjust_contrast_help_menu()
  % Test that the Adjust Contrast tool's Help menu holds only the tool's own
  % help item.  imcontrast_kb() used to call iptstandardhelp(), MathWorks'
  % helper that appends "Image Processing Toolbox Help", "Demos" and "About
  % Image Processing Toolbox" items; those open MathWorks' documentation and
  % demo browser rather than anything of APT's, and the About box is the
  % toolbox's easter egg.

  targetFigure = figure('Visible', 'off') ;
  cleanupTarget = onCleanup(@()(delete(targetFigure))) ;
  targetAxes = axes('Parent', targetFigure) ;
  imshow(uint8(repmat(0:4:252, [64 1])), 'Parent', targetAxes) ;
  toolFigure = imcontrast_kb(targetAxes) ;
  cleanupTool = onCleanup(@()(delete(toolFigure))) ;

  helpMenu = findall(toolFigure, 'Type', 'uimenu', 'Tag', 'help menu') ;
  assert(isscalar(helpMenu), 'Expected one Help menu on the Adjust Contrast tool') ;
  labels = get(allchild(helpMenu), {'Label'}) ;
  expectedLabels = {getString(message('images:imcontrastUIString:adjustContrastHelpMenubarLabel'))} ;
  assert(isequal(labels, expectedLabels), ...
         'The Help menu should hold only the Adjust Contrast help item, got: %s', ...
         strjoin(labels, ', ')) ;
end  % function
