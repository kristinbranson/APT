function s = idStream(initial_id)
%idStream ID Stream.
%   s = idStream(initial_id) returns an "identifier stream" as a structure
%   containing two function handles: nextId and recycleId.  An "id" is simply
%   an integer scalar.  If initial_id is omitted, it defaults to 1.
%  
%   s.nextId() returns the next available identifier.
%  
%   s.recycleId(id) is used to make a previously returned identifier
%   available to be used again.  If recycled identifiers are available,
%   s.nextId() returns the smallest one.
%
%   See also idStreamFactory.

%   Copyright 2005 The MathWorks, Inc.
%   $Revision $  $Date: 2005/05/27 14:07:20 $

default_initial_id = 1;
if nargin < 1
   initial_id = default_initial_id;
end

next_id = initial_id;
recycled_id_list = [];

s.nextId = @nextId;
s.recycleId = @recycleId;

   function id = nextId()
      [min_recycled_id, index] = min(recycled_id_list);
      use_recycled_id = ~isempty(min_recycled_id) && ...
         min_recycled_id < next_id;
      
      if use_recycled_id
         id = min_recycled_id;
         recycled_id_list(index) = [];
      else
         id = next_id;
         next_id = next_id + 1;
      end
   end

   function recycleId(id)
      recycled_id_list(end + 1) = id;
   end

end