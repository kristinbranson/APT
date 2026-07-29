function test_track_dialogs_known_num_animals()
% Exercise the "known number of animals" controls in the two multi-animal
% tracking dialogs (SpecifyMovieToTrackController and TrackBatchGUIController).
%
% These controls (a checkbox plus a numeric field) were ported from
% feature/idtracking during the main merge.  They appear only for MA projects,
% become enabled when identity linking is selected (and, for the batch dialog,
% when "maintain identities" is also checked), and their callbacks write
% id_known_num_animals / id_num_animals into the dialog's model struct.  This
% test opens each dialog on a real MA project, drives the controls exactly as
% the widget callbacks would, and asserts the enable-cascade and model updates.

% A bottom-up multi-animal project (MultiAnimal, no trx -> maIsMA is true),
% which is the case for which the identity-linking / known-num-animals controls
% are shown.
linuxProjectFilePath = ...
  ['/groups/branson/bransonlab/apt/unittest/' ...
   'htflies-10-with-saved-tracks-relocated.lbl'] ;
[projectFilePath, replacePath] = localize_test_project_path(linuxProjectFilePath) ;
[labeler, controller] = StartAPT('projfile', projectFilePath, ...
                                 'replace_path', replacePath) ;
cleaner = onCleanup(@()(delete(controller))) ;  %#ok<NASGU>

assert(labeler.maIsMA, 'Test project is not multi-animal; cannot exercise known-num-animals controls') ;

% ---------------------------------------------------------------------------
% SpecifyMovieToTrackController (Track > Current Movie).
% ---------------------------------------------------------------------------
mIdx = labeler.currMovIdx ;
movdata = struct() ;
movdata.movfiles = labeler.getMovieFilesAllFullMovIdx(mIdx) ;

sub = SpecifyMovieToTrackController(labeler, controller, movdata) ;
subCleaner = onCleanup(@()(delete(sub))) ;  %#ok<NASGU>

chk = sub.chk_known_num_animals ;
edt = sub.edit_num_animals ;
assert(~isempty(chk) && isgraphics(chk), 'SpecifyMovieToTrack: known-num-animals checkbox not created') ;
assert(~isempty(edt) && isgraphics(edt), 'SpecifyMovieToTrack: number-of-animals edit field not created') ;

% Linking defaults to motion, so the checkbox starts disabled.
assert(~enableIsOn_(chk), 'SpecifyMovieToTrack: checkbox should be disabled under motion linking') ;

% Switch to identity linking (popupmenu value 2) and fire its callback.
set(sub.pum_linking, 'Value', 2) ;
feval(sub.pum_linking.Callback, sub.pum_linking, []) ;
assert(strcmp(sub.link_type, 'identity'), 'SpecifyMovieToTrack: link_type did not become identity') ;
assert(enableIsOn_(chk), 'SpecifyMovieToTrack: checkbox should be enabled under identity linking') ;
assert(~enableIsOn_(edt), 'SpecifyMovieToTrack: edit field should stay disabled until checkbox is checked') ;

% Check "known number of animals": the edit field should become enabled and
% the model flag should be set.
set(chk, 'Value', 1) ;
feval(chk.Callback, chk, []) ;
assert(sub.movdata.id_known_num_animals, 'SpecifyMovieToTrack: id_known_num_animals not set true') ;
assert(enableIsOn_(edt), 'SpecifyMovieToTrack: edit field should be enabled once known-num is checked') ;

% Enter a number of animals.
set(edt, 'String', '7') ;
feval(edt.Callback, edt, []) ;
assert(isequal(sub.movdata.id_num_animals, 7), ...
       'SpecifyMovieToTrack: id_num_animals not recorded as 7 (got %s)', mat2str(sub.movdata.id_num_animals)) ;

clear subCleaner ;  % tear the movie-details dialog down before the batch part

% ---------------------------------------------------------------------------
% TrackBatchGUIController (Track > Track Multiple Videos).
% ---------------------------------------------------------------------------
controller.menu_track_batch_track_actuated_([], []) ;
tb = controller.trackBatchGUIController_ ;
assert(isa(tb, 'TrackBatchGUIController') && isvalid(tb), 'The batch-tracking dialog was not created') ;

chkB = tb.chk_known_num_animals ;
edtB = tb.edit_num_animals ;
assert(~isempty(chkB) && isgraphics(chkB), 'TrackBatch: known-num-animals checkbox not created') ;
assert(~isempty(edtB) && isgraphics(edtB), 'TrackBatch: number-of-animals edit field not created') ;

% Select identity linking; this enables the "maintain identities" checkbox.
set(tb.popup_linking, 'Value', 'identity') ;
feval(tb.popup_linking.ValueChangedFcn, tb.popup_linking, []) ;
assert(strcmp(tb.toTrack.link_type, 'identity'), 'TrackBatch: link_type did not become identity') ;
assert(enableIsOn_(tb.chk_maintain_identities), 'TrackBatch: maintain-identities checkbox should be enabled') ;

% Known-num-animals stays disabled until "maintain identities" is checked.
assert(~enableIsOn_(chkB), 'TrackBatch: known-num checkbox should be disabled until maintain-identities is checked') ;

% Check "maintain identities".
set(tb.chk_maintain_identities, 'Value', true) ;
feval(tb.chk_maintain_identities.ValueChangedFcn, tb.chk_maintain_identities, []) ;
assert(tb.toTrack.id_maintain_identity, 'TrackBatch: id_maintain_identity not set true') ;
assert(enableIsOn_(chkB), 'TrackBatch: known-num checkbox should be enabled once maintain-identities is checked') ;
assert(~enableIsOn_(edtB), 'TrackBatch: edit field should stay disabled until known-num is checked') ;

% Check "known number of animals" and enter a count.
set(chkB, 'Value', true) ;
feval(chkB.ValueChangedFcn, chkB, []) ;
assert(tb.toTrack.id_known_num_animals, 'TrackBatch: id_known_num_animals not set true') ;
assert(enableIsOn_(edtB), 'TrackBatch: edit field should be enabled once known-num is checked') ;

set(edtB, 'Value', 5) ;
feval(edtB.ValueChangedFcn, edtB, []) ;
assert(isequal(tb.toTrack.id_num_animals, 5), ...
       'TrackBatch: id_num_animals not recorded as 5 (got %s)', mat2str(tb.toTrack.id_num_animals)) ;

fprintf('test_track_dialogs_known_num_animals: PASSED\n') ;

end  % function

function tf = enableIsOn_(h)
% True if the widget's Enable state is 'on', for both uicontrol (char) and
% uifigure (OnOffSwitchState) components.
tf = strcmp(char(h.Enable), 'on') ;
end  % function
