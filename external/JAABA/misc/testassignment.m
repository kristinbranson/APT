function [samecost, sameassignment] = testassignment

%TESTASSIGN    Test and compare the assignment algorithms
%		[SAMECOST, SAMEASSIGNMENT] = TESTASSIGN randomly generates distance
%		matrices and solves the assignment problem using different algorithms.
%		Edit the header of this file to change the simulation parameters.
%
%		Written by Markus Buehren, www.Lss.uni-stuttgart.de
%		Last modified 14.12.2004

% simulation time in seconds
testTime = 10;       

% maximum matrix dimensions
maxOrders = [10, 20];  

% If infAllowed in set to false, only distance matrices without infinite 
%	costs are used
infAllowed = true;     

% If the product of the dimensions of the distance matrix is smaller than
% maxDimProduct, the assignment method computing all possible assignments
% is used as reference. Set this to inf to use it always or to 0 to never
% use.
maxDimProduct = 50;      

% start profiler
profile clear
profile on

% initialize
startTime = clock;
nassign = 0;
h = waitbar(0, 'Please wait');

while 1
	for dim1 = 1:maxOrders(1)
		for dim2 = 1:maxOrders(2)
			
			if ~infAllowed
				
				% generate distMatrix without infinite elements
				distMatrix = rand(dim1,dim2);
				
			else
				
				if rand(1) < 0.5
					% generate distMatrix with some infinite elements
					distMatrix = rand(dim1,dim2);
					
					if rand(1) < 0.5
						% set some elements to inf					
						distMatrix(find(rand(dim1,dim2) > rand(1))) = inf;
					end
					
				else
					
					% generate distMatrix with many infinite elements
					distMatrix = repmat(inf, dim1, dim2);
					
					if rand(1) < 0.5
						for row=1:dim1
							if rand(1) < 0.8
								% set one element per row to finite number
								distMatrix(row, 1+floor(dim2*rand(1))) = rand(1);
							end
							if rand(1) < 0.3
								% set another element per row to finite number
								distMatrix(row, 1+floor(dim2*rand(1))) = rand(1);
							end						
						end		
					else
						for col=1:dim2
							if rand(1) < 0.8
								% set one element per column to finite number
								distMatrix(1+floor(dim1*rand(1)), col) = rand(1);
							end
							if rand(1) < 0.3
								% set another element per column to finite number
								distMatrix(1+floor(dim1*rand(1)), col) = rand(1);
							end
						end		
					end
					
				end
			end
			
			% quantize distMatrix
			if rand(1) < 0.3
				distMatrix = round(max(10, 1000*rand(1)^2)*distMatrix);
			end			
			
			% transpose distMatrix
			if rand(1) < 0.5
				distMatrix = distMatrix';
			end
			
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			[assignmentCell{1}, costCell{1}] = assignmentoptimal(distMatrix);
			[assignmentCell{2}, costCell{2}] = assignmentsuboptimal1(distMatrix);
			[assignmentCell{3}, costCell{3}] = assignmentsuboptimal2(distMatrix);
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

			% if dimensions are moderate, compute all possible solutions
			if dim1 * dim2 < maxDimProduct
				[assignment, cost] = assignmentallpossible(distMatrix);
			else
				assignment = assignmentCell{1};
				cost       = costCell{1};
			end
			
			% count trial runs
			nassign = nassign + 1;
			
			if ~exist('samecost', 'var')
				M = length(costCell);
				samecost        = zeros(M,1);
				sameassignment = zeros(M,1);				
			end	
			
			% compute penalty for non-assignments
			finiteIndex = isfinite(distMatrix);
			penalty = max(max(distMatrix(finiteIndex))) * dim1 * dim2;
			if ~isempty(penalty)
				cost = cost + length(find(~assignment)) * penalty;
			end
			
			for m=1:M
				
				% penalize non-assignments
				if ~isempty(penalty)
					costCell{m} = costCell{m} + length(find(~assignmentCell{m})) * penalty;
				end				
				
				% compare costs
				if costCell{m} <= cost + 100*eps
					samecost(m) = samecost(m) + 1;
				end
				
				% compare assignments
				if all(assignmentCell{m} == assignment)
					sameassignment(m) = sameassignment(m) + 1;
				end
				
			end
		end
		% set waitbar
		curTime = etime(clock, startTime);
		waitbar(curTime/testTime, h);
	end
	
	% stop after given time
	if curTime > testTime
		break
	end
end
close(h);

% scale counters
samecost       = samecost/nassign;
sameassignment = sameassignment/nassign;

profile report

