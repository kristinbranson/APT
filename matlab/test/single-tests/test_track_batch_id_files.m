function test_track_batch_id_files()
% Exercise the two identity-linking file controls in the batch-tracking dialog
% (TrackBatchGUIController): the ID model file and the detected identities file.
%
% Both files let successive tracking runs share the same identities: the ID model
% is the identity classifier, and the detected identities are the ID cluster
% centers found with it.  The dialog has to record what the user picks, the
% setting has to survive the movie-list JSON round-trip, the Labeler has to
% remember the locations in the project file, and a remembered location has to
% prefill the dialog the next time an ID tracking job is set up.

% A bottom-up multi-animal project (MultiAnimal, no trx -> maIsMA is true),
% which is the case for which the identity-linking controls are shown.
linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'htflies-10-with-saved-tracks-relocated.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleaner = onCleanup(@()(delete(controller))) ;

assert(labeler.maIsMA, 'Test project is not multi-animal; cannot exercise the identity-linking file controls') ;

idModelFilePath = fullfile(tempdir(), 'test_id_wts.p') ;
detectedIdentitiesFilePath = fullfile(tempdir(), 'test_detected_identities.p') ;

% ---------------------------------------------------------------------------
% TrackBatchGUIController (Track > Track Multiple Videos).
% ---------------------------------------------------------------------------
controller.menu_track_batch_track_actuated_([], []) ;
trackBatchController = controller.trackBatchGUIController_ ;
assert(isa(trackBatchController, 'TrackBatchGUIController') && isvalid(trackBatchController), ...
       'The batch-tracking dialog was not created') ;

idModelEdit = trackBatchController.edit_id_model_file ;
idModelButton = trackBatchController.button_id_model_file ;
identitiesEdit = trackBatchController.edit_detected_identities_file ;
identitiesButton = trackBatchController.button_detected_identities_file ;
assert(~isempty(idModelEdit) && isgraphics(idModelEdit), 'TrackBatch: ID model file edit field not created') ;
assert(~isempty(idModelButton) && isgraphics(idModelButton), 'TrackBatch: ID model file browse button not created') ;
assert(~isempty(identitiesEdit) && isgraphics(identitiesEdit), 'TrackBatch: detected identities file edit field not created') ;
assert(~isempty(identitiesButton) && isgraphics(identitiesButton), 'TrackBatch: detected identities file browse button not created') ;

% Linking defaults to motion, so the controls start disabled.
assert(~enableIsOn_(idModelEdit), 'TrackBatch: ID model file field should be disabled under motion linking') ;
assert(~enableIsOn_(identitiesEdit), 'TrackBatch: detected identities file field should be disabled under motion linking') ;

% Select identity linking; this enables all four controls.
set(trackBatchController.popup_linking, 'Value', 'identity') ;
feval(trackBatchController.popup_linking.ValueChangedFcn, trackBatchController.popup_linking, []) ;
assert(strcmp(trackBatchController.toTrack.link_type, 'identity'), 'TrackBatch: link_type did not become identity') ;
assert(enableIsOn_(idModelEdit), 'TrackBatch: ID model file field should be enabled under identity linking') ;
assert(enableIsOn_(idModelButton), 'TrackBatch: ID model file browse button should be enabled under identity linking') ;
assert(enableIsOn_(identitiesEdit), 'TrackBatch: detected identities file field should be enabled under identity linking') ;
assert(enableIsOn_(identitiesButton), 'TrackBatch: detected identities browse button should be enabled under identity linking') ;

% Typing paths into the edit fields should record them in the dialog's model struct.
set(idModelEdit, 'Value', idModelFilePath) ;
feval(idModelEdit.ValueChangedFcn, idModelEdit, []) ;
assert(strcmp(trackBatchController.toTrack.id_model_file, idModelFilePath), ...
       'TrackBatch: id_model_file not recorded as %s (got %s)', ...
       idModelFilePath, trackBatchController.toTrack.id_model_file) ;

set(identitiesEdit, 'Value', detectedIdentitiesFilePath) ;
feval(identitiesEdit.ValueChangedFcn, identitiesEdit, []) ;
assert(strcmp(trackBatchController.toTrack.id_detected_identities_file, detectedIdentitiesFilePath), ...
       'TrackBatch: id_detected_identities_file not recorded as %s (got %s)', ...
       detectedIdentitiesFilePath, trackBatchController.toTrack.id_detected_identities_file) ;

% ---------------------------------------------------------------------------
% JSON round-trip: the settings have to survive save-and-reload of the movie list.
% ---------------------------------------------------------------------------
toTrack = trackBatchController.toTrack ;
toTrack.movfiles = labeler.getMovieFilesAllFullMovIdx(labeler.currMovIdx) ;
toTrack.trkfiles = {fullfile(tempdir(), 'test_track_batch_id_files.trk')} ;
toTrack.trxfiles = {''} ;
toTrack.cropRois = {[]} ;
toTrack.calibrationfiles = {''} ;
toTrack.targets = {[]} ;
toTrack.f0s = {1} ;
toTrack.f1s = {inf} ;
jsonFilePath = [tempname() '.json'] ;
jsonCleaner = onCleanup(@()(deleteIfExists_(jsonFilePath))) ;
writeToTrackJSON(toTrack, jsonFilePath) ;
reloadedToTrack = parseToTrackJSON(jsonFilePath, labeler) ;
assert(isfield(reloadedToTrack, 'id_model_file') && strcmp(reloadedToTrack.id_model_file, idModelFilePath), ...
       'The ID model file did not survive the toTrack JSON round-trip') ;
assert(isfield(reloadedToTrack, 'id_detected_identities_file') && ...
       strcmp(reloadedToTrack.id_detected_identities_file, detectedIdentitiesFilePath), ...
       'The detected identities file did not survive the toTrack JSON round-trip') ;

% ---------------------------------------------------------------------------
% The Labeler remembers the locations, and they are saved with the project.
% ---------------------------------------------------------------------------
labeler.rememberIdentityTrackingFiles(idModelFilePath, detectedIdentitiesFilePath) ;
assert(strcmp(labeler.idModelFile, idModelFilePath), 'The Labeler did not remember the ID model file') ;
assert(strcmp(labeler.idDetectedIdentitiesFile, detectedIdentitiesFilePath), ...
       'The Labeler did not remember the detected identities file') ;
saveStruct = labeler.projGetSaveStruct() ;
assert(isfield(saveStruct, 'idModelFile') && strcmp(saveStruct.idModelFile, idModelFilePath), ...
       'The ID model file is not saved with the project') ;
assert(isfield(saveStruct, 'idDetectedIdentitiesFile') && ...
       strcmp(saveStruct.idDetectedIdentitiesFile, detectedIdentitiesFilePath), ...
       'The detected identities file is not saved with the project') ;

% ---------------------------------------------------------------------------
% A newly opened dialog prefills both fields from what the project remembers.
% ---------------------------------------------------------------------------
controller.deleteTrackBatchGUIController() ;
controller.menu_track_batch_track_actuated_([], []) ;
freshController = controller.trackBatchGUIController_ ;
assert(strcmp(freshController.toTrack.id_model_file, idModelFilePath), ...
       'The batch dialog did not prefill the remembered ID model file (got %s)', ...
       freshController.toTrack.id_model_file) ;
assert(strcmp(freshController.toTrack.id_detected_identities_file, detectedIdentitiesFilePath), ...
       'The batch dialog did not prefill the remembered detected identities file (got %s)', ...
       freshController.toTrack.id_detected_identities_file) ;
assert(strcmp(freshController.edit_id_model_file.Value, idModelFilePath), ...
       'The batch dialog ID model file field does not show the remembered location') ;
assert(strcmp(freshController.edit_detected_identities_file.Value, detectedIdentitiesFilePath), ...
       'The batch dialog detected identities field does not show the remembered location') ;

% ---------------------------------------------------------------------------
% ToTrackInfo, which is what the command generation reads the settings from.
% A user-chosen ID model file must survive setDefaultFiles(), which otherwise
% derives the ID model location from the output trk file.
% ---------------------------------------------------------------------------
totrackinfo = ToTrackInfo('movfiles', {'/path/to/movie.avi'}, ...
                          'trkfiles', {'/path/to/movie.trk'}, ...
                          'views', 1, ...
                          'stages', 1, ...
                          'link_type', 'identity', ...
                          'id_detected_identities_file', detectedIdentitiesFilePath, ...
                          'id_model_file_requested', idModelFilePath) ;
assert(strcmp(totrackinfo.getIDDetectedIdentitiesFile(), detectedIdentitiesFilePath), ...
       'ToTrackInfo did not hold on to the detected identities file') ;
totrackinfo.setDefaultIDModelFile() ;
assert(strcmp(totrackinfo.getIDModelFile(), idModelFilePath), ...
       'ToTrackInfo.setDefaultIDModelFile() overwrote the user-chosen ID model file (got %s)', ...
       totrackinfo.getIDModelFile()) ;

% Without a user-chosen file, the ID model location is still derived from the trk file.
defaultTotrackinfo = ToTrackInfo('movfiles', {'/path/to/movie.avi'}, ...
                                 'trkfiles', {'/path/to/movie.trk'}, ...
                                 'views', 1, ...
                                 'stages', 1) ;
defaultTotrackinfo.setDefaultIDModelFile() ;
assert(strcmp(defaultTotrackinfo.getIDModelFile(), fullfile('/path/to', 'id_wts_movie.p')), ...
       'The default ID model file location changed (got %s)', defaultTotrackinfo.getIDModelFile()) ;

end  % function

function tf = enableIsOn_(h)
% True if the widget's Enable state is 'on', for both uicontrol (char) and
% uifigure (OnOffSwitchState) components.
tf = strcmp(char(h.Enable), 'on') ;
end  % function

function deleteIfExists_(filePath)
% Delete filePath if it is there, doing nothing if it is not.
if exist(filePath, 'file')
  delete(filePath) ;
end
end  % function
