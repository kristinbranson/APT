function ndx = sub2indv(siz,v,varargin)
%SUB2INDV Linear index from multiple subscripts.
%   SUB2INDV is used to determine the equivalent single index
%   corresponding to a given set of subscript values.
%
%   IND = SUB2IND(SIZ,V) returns the linear index
%   equivalent to the N subscripts in the arrays V = [I1,I2,...,IN] for an
%   array of size SIZ.
%
%   IND will have the same number of rows as v. 
%
%   Class support for inputs I,J: 
%      float: double, single
%      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
%

if ndims(v) > 2, %#ok<ISMAT>
  error('V must have at most 2 dimensions');
end
siz = double(siz);

numOfIndInput = length(siz);
if numOfIndInput < 2
    error(message('MATLAB:sub2indv:InvalidSize'));
end

if any(min(v(:,1)) < 1) || any(max(v(:,1)) > siz(1))
    %Verify subscripts are within range
    error(message('MATLAB:sub2indv:IndexOutOfRange'));
end

ndx = double(v(:,1));
s = size(v(:,1));
if numOfIndInput >= 2
    if any(min(v(:,2)) < 1) || any(max(v(:,2)) > siz(2))
        %Verify subscripts are within range
        error(message('MATLAB:sub2ind:IndexOutOfRange'));
    end
    %Compute linear indices
    ndx = ndx + (double(v(:,2)) - 1).*siz(1);
end 
    
if numOfIndInput > 2
    %Compute linear indices
    k = cumprod(siz);
    for i = 3:numOfIndInput
        if (any(min(v(:,i)) < 1)) || (any(max(v(:,i)) > siz(i)))
            %Verify subscripts are within range
            error(message('MATLAB:sub2ind:IndexOutOfRange'));
        end
        ndx = ndx + (double(v(:,i))-1)*k(i-1);
    end
end