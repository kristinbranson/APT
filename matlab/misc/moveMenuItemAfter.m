function moveMenuItemAfter(hnew,hprev)

hpar = get(hprev,'Parent');
assert(get(hnew,'Parent') == hpar);
newpos = min(get(hprev,'Position') + 1,numel(get(hpar,'Children')));
set(hnew,'Position',newpos);
