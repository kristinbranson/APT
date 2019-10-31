function [cpstruct] = cpsave(varNames,inputBasePairsVector)
% CPSAVE Save control points.
%   CPSAVE is called by the Java class SaveToWorkspace to save control points.
%
%   varNames is a cell array that contains the variable names for
%   input_points, base_points, cpstruct. 
%   varNames = { input_str, base_str, cpstruct_str}
%   If any of the strings in varNames are empty, then the corresponding
%   variable is not saved to the workspace.
%
%   inputBasePairsVector is a java.util.Vector that stores a vector of
%   ControlPointPair objects. From inputBasePairsVector, we can extract
%   variables to save valid pairs or all points to the workspace.

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
 
%   Copyright 1993-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.6 $  $Date: 2011/07/19 23:57:36 $

% validate input
narginchk(2,2)

if ~isa(inputBasePairsVector,'java.util.Vector')
    error(message('images:cpsave:invalidInputBasePairsVector'))
end

% varNames should be a cell array of strings
if ~iscell(varNames)
    error(message('images:cpsave:varNamesNotCellArray'))
end

cpstruct = [];

npairs = inputBasePairsVector.size;

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

for i = 0:(npairs-1)

    pair = inputBasePairsVector.elementAt(i);
    ids(i+1) = pair.getPairId;
    
    inputPoint = pair.getInputPoint;
    if ~isempty(inputPoint)
        i_inputPoints = i_inputPoints + 1;
        inputPoints(i_inputPoints,:) = [inputPoint.x inputPoint.y];
        inputIdPairs(i_inputPoints,:) = [i_inputPoints (i+1)];
        isInputPredicted(i_inputPoints) = pair.isPredicted(inputPoint);
    end
    
    basePoint = pair.getBasePoint;
    if ~isempty(basePoint)
        i_basePoints = i_basePoints + 1;
        basePoints(i_basePoints,:) = [basePoint.x basePoint.y];
        baseIdPairs(i_basePoints,:) = [i_basePoints (i+1)];
        isBasePredicted(i_basePoints) = pair.isPredicted(basePoint);
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
cpstruct.inputPoints = inputPoints + 1; % add 1 to account for
                                        % zero-based Java coordinates 
cpstruct.basePoints = basePoints + 1;
cpstruct.inputBasePairs = inputBasePairs;
cpstruct.ids = ids;
cpstruct.inputIdPairs = inputIdPairs;
cpstruct.baseIdPairs = baseIdPairs;
cpstruct.isInputPredicted = isInputPredicted;
cpstruct.isBasePredicted = isBasePredicted;

% convert to pairs
[input_points,base_points] = cpstruct2pairs(cpstruct);

if ~isempty(varNames{1})
    assignin('base',varNames{1},input_points);
end

if ~isempty(varNames{2})
    assignin('base',varNames{2},base_points);
end

if ~isempty(varNames{3})
    assignin('base',varNames{3},cpstruct);
end
