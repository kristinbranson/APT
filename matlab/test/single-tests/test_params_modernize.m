function test_params_modernize()
  % Test that APTParameters.modernize migrates parameter structs saved by
  % older APT versions onto the current parameter tree: the MultiAnimal
  % Detect subtree moves to MultiAnimalDetect, animal-count limits move
  % under Track, the TargetCrop radius and TrackletStitch subtrees move to
  % their current homes, shared DeepTrack parameters move to
  % DeepTrackShared, and the removed ImageProcessing subtree is dropped.

  old = struct;
  old.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius = 90;
  old.ROOT.ImageProcessing.BackSub.Use = false;
  old.ROOT.MultiAnimal.Detect.max_n_animals = 7;
  old.ROOT.MultiAnimal.Detect.min_n_animals = 2;
  old.ROOT.MultiAnimal.Detect.BBox.MinAspectRatio = 0.5;
  old.ROOT.MultiAnimal.TrackletStitch.link_stage = 'first';
  old.ROOT.MultiAnimal.TrackletStitch.link_id = true;
  old.ROOT.MultiAnimal.TrackletStitch.link_id_cropsz = 144;
  old.ROOT.DeepTrack.DataAugmentation.rrange = 25;
  old.ROOT.DeepTrack.ImageProcessing.imax = 128;

  capturedOutput = evalc('new = APTParameters.modernize(old) ;') ;
  assert(~contains(capturedOutput, 'Ignoring unrecognized field'), ...
         'modernize() left unrecognized fields for structoverlay to warn about:\n%s', ...
         capturedOutput) ;

  assert(new.ROOT.MultiAnimal.TargetCrop.ManualRadius == 90, ...
         'TargetCrop.Radius should migrate to TargetCrop.ManualRadius');
  assert(~isfield(new.ROOT,'ImageProcessing'), ...
         'The removed ImageProcessing subtree should be dropped');
  assert(new.ROOT.MultiAnimal.Track.max_n_animals == 7, ...
         'Detect.max_n_animals should migrate to Track.max_n_animals');
  assert(new.ROOT.MultiAnimal.Track.min_n_animals == 2, ...
         'Detect.min_n_animals should migrate to Track.min_n_animals');
  assert(new.ROOT.MultiAnimal.Track.max_n_animals_user == 7, ...
         'max_n_animals_user should be seeded from the project max_n_animals');
  assert(new.ROOT.MultiAnimalDetect.BBox.MinAspectRatio == 0.5, ...
         'MultiAnimal.Detect should migrate to MultiAnimalDetect');
  assert(strcmp(new.ROOT.MultiAnimal.Track.TrackletStitch.link_stage,'first'), ...
         'TrackletStitch should migrate under Track');
  assert(new.ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_height == 144 && ...
         new.ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_width == 144, ...
         'link_id_cropsz should migrate to link_id_cropsz_height/width');
  assert(~isfield(new.ROOT.MultiAnimal.Track.TrackletStitch,'link_id_cropsz'), ...
         'The scalar link_id_cropsz should be gone after migration');
  assert(~isfield(new.ROOT.MultiAnimal.Track.TrackletStitch,'link_id'), ...
         'The retired link_id parameter should be dropped');
  assert(new.ROOT.DeepTrack.DataAugmentation.rrange == 25, ...
         'Pose-stage DataAugmentation values should be preserved');
  assert(new.ROOT.DeepTrackShared.ImageProcessing.imax == 128, ...
         'Shared DeepTrack parameters should migrate to DeepTrackShared');
end  % function
