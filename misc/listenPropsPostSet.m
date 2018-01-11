function lstnrObj = listenPropsPostSet(obj,props,cbk)
% Listen to multiple properties with a single callback
%
% obj: scalar handle obj to be listened to
% props: cellstr, properties to listen to
% cbk: callback
%
% lstnrObj: scalar listener object, lifecycle not tied to obj

assert(iscellstr(props) && isvector(props));
assert(isa(cbk,'function_handle'));

mcls = metaclass(obj);
mprops = mcls.PropertyList;
[tf,loc] = ismember(props,{mprops.Name}');
iNotFound = find(~tf);
for i=iNotFound(:)'
  warningNoTrace('listenprops:prop','Property ''%s'' not found.',props{i});
end
mprops = mprops(loc(tf));
lstnrObj = event.proplistener(obj,mprops,'PostSet',cbk);
