function test_occluded_box_point_tags()
  % Showing the occluded-points box must not reuse the main points' Tag.
  %
  % LabelCoreController.showOcc() creates, in the occluded-box axes, a marker
  % tagged LabelCore_PtsOcc_<i> and a number text for each point.  The text
  % was tagged LabelCore_Pts_<i>, which is the Tag of the main axes' point
  % marker and of its number text, so findall(0, 'Tag', 'LabelCore_Pts_1',
  % 'Type', 'text') answered two objects -- and since findall lists the
  % newest first, a tool that reads "the" point text by Tag read the
  % occluded-box one, at NaN (found by the Python port's side-by-side
  % comparison, 2026-10-01).  The occ text takes the occ marker's Tag.

  linux_project_file_path = ...
    '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  if labeler.showOccludedBox
    controller.controlActuated('menu_view_occluded_points_box') ;  % hide it first, so the toggle below shows it
    drawnow() ;
  end
  controller.controlActuated('menu_view_occluded_points_box') ;
  drawnow() ;
  assert(labeler.showOccludedBox, 'The occluded-points box did not show') ;

  pointCount = labeler.nLabelPoints ;
  for pointIndex = 1 : pointCount
    mainTag = sprintf('LabelCore_Pts_%d', pointIndex) ;
    occTag = sprintf('LabelCore_PtsOcc_%d', pointIndex) ;

    mainTexts = findall(0, 'Tag', mainTag, 'Type', 'text') ;
    assert(numel(mainTexts) == 1, ...
           'Expected one text tagged %s, found %d', mainTag, numel(mainTexts)) ;
    assert(strcmp(get(ancestor(mainTexts, 'axes'), 'Tag'), 'axes_curr'), ...
           'The text tagged %s is not in the main axes', mainTag) ;

    occTexts = findall(0, 'Tag', occTag, 'Type', 'text') ;
    assert(numel(occTexts) == 1, ...
           'Expected one text tagged %s in the occluded box, found %d', occTag, numel(occTexts)) ;
    occMarkers = findall(0, 'Tag', occTag, 'Type', 'line') ;
    assert(numel(occMarkers) == 1, ...
           'Expected one marker tagged %s in the occluded box, found %d', occTag, numel(occMarkers)) ;
  end
end  % function
