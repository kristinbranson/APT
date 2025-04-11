function result = dir_without_links(template)
    ds0 = dir(template) ;  % ds for "dir struct", nx1, includes '.' and '..' entries
    ds1 = ds0(3:end) ;  % trim '.' and '..' entries
    is_link_from_raw_index = cellfun(@isempty, {ds1.bytes}') ;
    result = ds1(~is_link_from_raw_index) ;
end
