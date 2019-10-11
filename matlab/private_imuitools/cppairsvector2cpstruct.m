function [cpstruct] = cppairsvector2cpstruct(inputBasePairsVector)
% CPPAIRSVECTOR2CPSTRUCT Save control points.
%   CPSAVE is called by CPSELECT to save control points.
%
%   inputBasePairsVector is a structure array that stores control point pair
%   information. From inputBasePairsVector, we can extract variables to save
%   valid pairs or all points to the workspace.

% cpstruct fields:
%
% inputPoints      M-by-N  coordinates of Input points, M points, N dimensions
%
% basePoints       R-by-N  coordinates of Base points, R points, N dimensions
%
% inputBasePairs   P-by-2  correspondence between xyInput and xyBase, 
%                          P pairs of indices into xyInput and xyBase
%
% ids              Q-by-1  numeric ids for points, Q independent ids, 
%                          both members of a pair share the same id
%
% inputIdPairs     K-by-2  correspondence between xyInput and id, 
%                          K pairs of indices, K=M (unless some points lack ids)
%
% baseIdPairs      L-by-2  correspondence between xyBase and id, 
%                          L pairs of indices, L=R (unless some points lack ids)
%
% isInputPredicted M-by-1  Flag on each member of xyInput, 0 if not predicted, 1 if predicted
%
% isBasePredicted  R-by-1  Flag on each member of xyBase, 0 if not predicted, 1 if predicted
 
%   Copyright 1993-2006 The MathWorks, Inc.
%   $Revision $  $Date: 2006/06/15 20:10:09 $


cpstruct = [];

npairs = length(inputBasePairsVector);

inputPoints = zeros(npairs,2);
basePoints = zeros(npairs,2);
inputBasePairs = zeros(npairs,2);
ids = zeros(npairs,1);
inputIdPairs = zeros(npairs,2);
baseIdPairs = zeros(npairs,2);
isInputPredicted = zeros(npairs,1);
isBasePredicted = zeros(npairs,1);

i_inputPoints = 0;
i_basePoints = 0;
i_inputBasePairs = 0;

for i = 0:npairs-1

  pair = inputBasePairsVector(i+1);
  ids(i+1) = pair.id;
  
  % Function to use below to check if either the inputPoint or the basePoint
  % is predicted.
  isPredicted = @(p) isequal(p,pair.predictedPoint);
  
  inputPoint = pair.inputPoint;
  if ~isempty(inputPoint)
    inputPointAPI = iptgetapi(inputPoint.hDetailPoint);      
    i_inputPoints = i_inputPoints + 1;
    inputPoints(i_inputPoints,:) = inputPointAPI.getPosition();
    inputIdPairs(i_inputPoints,:) = [i_inputPoints (i+1)];
    isInputPredicted(i_inputPoints) = isPredicted(inputPoint);
  end
  
  basePoint = pair.basePoint;
  if ~isempty(basePoint)
    basePointAPI = iptgetapi(basePoint.hDetailPoint);          
    i_basePoints = i_basePoints + 1;
    basePoints(i_basePoints,:) = basePointAPI.getPosition();
    baseIdPairs(i_basePoints,:) = [i_basePoints (i+1)];
    isBasePredicted(i_basePoints) = isPredicted(basePoint);
  end
  
  if ( ~isempty(inputPoint) && ~isempty(basePoint) )
    i_inputBasePairs = i_inputBasePairs + 1;
    inputBasePairs(i_inputBasePairs,:) = [i_inputPoints i_basePoints];
  end

end

% truncate unused space
inputPoints(i_inputPoints+1:npairs,:) = [];
basePoints(i_basePoints+1:npairs,:) = [];
inputBasePairs(i_inputBasePairs+1:npairs,:) = [];
inputIdPairs(i_inputPoints+1:npairs,:) = [];
baseIdPairs(i_basePoints+1:npairs,:) = [];
isInputPredicted(i_inputPoints+1:npairs,:) = [];
isBasePredicted(i_basePoints+1:npairs,:) = [];

% fill cpstruct
cpstruct.inputPoints = inputPoints;
cpstruct.basePoints = basePoints;
cpstruct.inputBasePairs = inputBasePairs;
cpstruct.ids = ids;
cpstruct.inputIdPairs = inputIdPairs;
cpstruct.baseIdPairs = baseIdPairs;
cpstruct.isInputPredicted = isInputPredicted;
cpstruct.isBasePredicted = isBasePredicted;
