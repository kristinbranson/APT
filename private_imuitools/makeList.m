function api = makeList
%MAKELIST Make a list structure with API functions.
%   api = makeList returns a structure containing function handles for
%   manipulating a list.
%
%   id = api.appendItem(new_item) appends new_item to the end of the
%   list.  The output value, id, is only used if you want to remove an
%   item from the list.
%
%   api.removeItem(id) removes the item with the corresponding id value
%   from the list.  If there is no item with the corresponding id value,
%   api.removeItem(id) just returns silently.
%
%   list = api.getList() returns the entire list as a struct array where
%   each element is a structure with the following two fields:
%
%      Item : the list item added with the appendItem function
%      ID   : a unique integer id for the list item.  IDs are not reused.
%
%   Note: The current simple implementation does not perform the append and
%   remove operations as efficiently as a true linked list.  It is not
%   suitable for applications that need to append to and remove from lists
%   containing a large number of items.

%   Copyright 2004-2008 The MathWorks, Inc.
%   $Revision: 1.1.8.2 $  $Date: 2008/01/15 18:53:32 $

list = struct('ID', {}, 'Item', {});
next_available_id = 0;

api.appendItem = @appendItem;
api.removeItem = @removeItem;
api.getList    = @getList;

  function id = appendItem(item)
    id = next_available_id;
    next_available_id = next_available_id + 1;
    list(end + 1).ID = id;
    list(end).Item = item;
  end
  
  function removeItem(id)
    for k = 1:numel(list)
      if list(k).ID == id
        list(k) = [];
        break;
      end
    end
  end
  
  function out = getList
    out = list;
  end

end
