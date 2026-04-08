classdef TrkFileMock
  % Minimal mock for confidenceBoutsFromTrkFile() tests.

  properties
    ntracklets
    pTrkiTgt
    frameIndicesFromTrackletIndex
    periodFromTrackletIndex
    phaseOffsetFromTrackletIndex
    amplitudeFromTrackletIndex
    offsetFromTrackletIndex
    landmarkCount
    isConfidenceLackThereof
  end

  methods
    function obj = TrkFileMock(trackletSpecs, landmarkCount, isConfidenceLackThereof)
      if nargin < 2 || isempty(landmarkCount)
        landmarkCount = 3 ;
      end
      if nargin < 3 || isempty(isConfidenceLackThereof)
        isConfidenceLackThereof = false ;
      end

      obj.ntracklets = numel(trackletSpecs) ;
      obj.pTrkiTgt = reshape([trackletSpecs.targetIndex], [], 1) ;
      obj.frameIndicesFromTrackletIndex = {trackletSpecs.frameIndices}' ;
      obj.periodFromTrackletIndex = reshape([trackletSpecs.period], [], 1) ;
      obj.phaseOffsetFromTrackletIndex = reshape([trackletSpecs.phaseOffset], [], 1) ;
      obj.amplitudeFromTrackletIndex = reshape([trackletSpecs.amplitude], [], 1) ;
      obj.offsetFromTrackletIndex = reshape([trackletSpecs.offset], [], 1) ;
      obj.landmarkCount = landmarkCount ;
      obj.isConfidenceLackThereof = isConfidenceLackThereof ;
    end

    function [xy, occ, fr, aux] = getPTrkTgt(obj, trackletIndex, varargin)
      auxflds = {} ;
      for i = 1 : 2 : numel(varargin)
        if strcmp(varargin{i}, 'auxflds')
          auxflds = varargin{i + 1} ;
        end
      end

      fr = obj.frameIndicesFromTrackletIndex{trackletIndex} ;
      frameCount = numel(fr) ;
      xy = nan(obj.landmarkCount, 2, frameCount, 1) ;
      occ = false(obj.landmarkCount, frameCount, 1) ;

      if isempty(auxflds)
        aux = [] ;
      elseif isequal(auxflds, {'pTrkConf'})
        conf = obj.confidenceFromTrackletIndex_(trackletIndex) ;
        aux = reshape(conf, obj.landmarkCount, frameCount, 1, 1) ;
      else
        error('TrkFileMock:UnsupportedAuxField', ...
              'Unsupported aux field request in TrkFileMock.') ;
      end
    end
  end

  methods (Access = private)
    function conf = confidenceFromTrackletIndex_(obj, trackletIndex)
      frameCount = numel(obj.frameIndicesFromTrackletIndex{trackletIndex}) ;
      period = obj.periodFromTrackletIndex(trackletIndex) ;
      halfPeriod = period / 2 ;
      if abs(halfPeriod - round(halfPeriod)) > eps(period)
        error('TrkFileMock:InvalidPeriod', ...
              'Triangle-wave period must be even.') ;
      end

      phaseOffset = obj.phaseOffsetFromTrackletIndex(trackletIndex) ;
      framePhaseOffsetFromFrameIndex = (0 : frameCount - 1)' + phaseOffset ;
      landmarkPhaseOffsetFromLandmarkIndex = 0 : obj.landmarkCount - 1 ;
      phase = mod(framePhaseOffsetFromFrameIndex + landmarkPhaseOffsetFromLandmarkIndex, period) ;
      triangle = 1 - abs(phase / halfPeriod - 1) ;

      amplitude = obj.amplitudeFromTrackletIndex(trackletIndex) ;
      offset = obj.offsetFromTrackletIndex(trackletIndex) ;
      conf = offset + amplitude * triangle' ;
      if obj.isConfidenceLackThereof
        conf = -conf ;
      end
    end
  end
end
