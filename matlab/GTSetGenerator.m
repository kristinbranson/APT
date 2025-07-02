classdef GTSetGenerator < handle
  % GTSetGenerator holds a "GT set specification" and can generate a GT set
  % per that specification.
  %
  % A GT Set Specification is a description of how to generate a random GT
  % set: use so-and-so-movies, generate so many frames in a certain manner,
  % etc.
  %
  % A GT Set is a particular instantiation of a GT Set Specification: an 
  % MFTable containing certain rows to be GT labeled.
  %
  % Note on numFrames and numFramesType. 
  % Together, the minDistTraining/mIdxs properties of GTSetGenerator 
  % specify a "base set" of available GT rows. numFrames and numFramesType 
  % specify a random sampling from those GT rows. 
  %
  % * If numFramesType is 'total', numFrames rows are drawn randomly from
  %   the base set. If some movies have more frames than others, those 
  %   movies will be overweighted per their relative frame numbers.
  %   Similary, if some targets in some movies have more frames, then they 
  %   will be overweighted etc.
  %
  % * If numFramesType is 'permovie', numFrames rows are drawn randomly
  %   from each movie in mIdxs. This ensures equal balancing across movies,
  %   regardless of movie lengths. No balancing is done on targets within a
  %   single movie (when applicable); targets wth more frames are
  %   overweighted, etc.
  %
  % * If numFramesType is 'pertarget', numFrames rows are drawn randomly
  %   for each (movie,target) pair. This ensures equal balancing across all
  %   targets, regardless of the number of rows available for each target.
    
  properties
    numFrames % positive int
    numFramesType % scalar GTSetNumFramesType
    minDistTraining % nonnegative int. 0 => no account of training frames is taken
    mIdxs % vector of movieindices
  end
  
  methods
  
    function obj = GTSetGenerator(nfrms,nftype,mindist,mIdxs)
      assert(isscalar(nfrms) && round(nfrms)==nfrms && nfrms>=1);
      assert(isscalar(nftype) && isa(nftype,'GTSetNumFramesType'));
      assert(isscalar(mindist) && round(mindist)==mindist && mindist>=0);
      assert(isa(mIdxs,'MovieIndex'));
      
      obj.numFrames = nfrms;
      obj.numFramesType = nftype;
      obj.minDistTraining = mindist;
      obj.mIdxs = mIdxs;
    end
    
    function tblGT = generate(obj,lObj)
      
      %wbObj = WaitBarWithCancel('Compiling GT frames');
      %oc = onCleanup(@()delete(wbObj));
      wbObj = lObj.progressMeter ;
      wbObj.arm('title', 'Compiling GT frames');
      oc = onCleanup(@()(lObj.disarmProgressMeter())) ;

      % Step 1: Generate base set of GT frames which will be randomly
      % sampled
      
      tfAvoid = obj.minDistTraining>0;
      tfBaseGenerated = false;
      if tfAvoid

        % (if nec) find "avoid table" of labeled frms in movies that are in
        % both regular and GT sets

        [iMovRegOverlap,iMovGTOverlap] = lObj.gtCommonMoviesRegGT();
        if ~isempty(iMovRegOverlap)
          mIdxReg = MovieIndex(iMovRegOverlap);
          mftset = MFTSet(...
            MovieIndexSetFixed(mIdxReg),...
            FrameSetVariable.LabeledFrm,...
            FrameDecimationFixed.EveryFrame,...
            TargetSetVariable.AllTgts);
          
          % labeled frames in reg movies that are in the GT list
          tblAvoid = mftset.getMFTable(lObj,'wbObj',wbObj);
          
          if ~isempty(tblAvoid)
            mapMovIdxRemap = containers.Map(...
              int32(iMovRegOverlap),-int32(iMovGTOverlap));
            tblAvoid = MFTable.remapIntegerKey(tblAvoid,'mov',mapMovIdxRemap);
            assert(isa(tblAvoid.mov,'MovieIndex'));

            fset = FrameSetVariable(@(lo)'',@(lo)'',...
              @FrameSetVariable.allFrmGetFrms,...
              'avoidTbl',tblAvoid,'avoidRadius',obj.minDistTraining);
            mftset = MFTSet(...
              MovieIndexSetFixed(obj.mIdxs),...
              fset,...
              FrameDecimationFixed.EveryFrame,...
              TargetSetVariable.AllTgts);

            tblBase = mftset.getMFTable(lObj,'wbObj',wbObj); 
            tfBaseGenerated = true;
          end
        end
      end
      
      if ~tfBaseGenerated
        % get base set of avail GT frms: all frms/tgts for obj.mIdxs
        mftset = MFTSet(...
          MovieIndexSetFixed(obj.mIdxs),...
          FrameSetVariable.AllFrm,...
          FrameDecimationFixed.EveryFrame,...
          TargetSetVariable.AllTgts);
        tblBase = mftset.getMFTable(lObj,'wbObj',wbObj);
      end
      
      % Step 2: Randomly sample tblBase per .numFrames*
      tblGT = obj.numFramesType.sample(tblBase,obj.numFrames);      
    end
    
  end
  
end