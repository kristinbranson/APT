function moveMenuItemBefore(hnew,hafter)

hpar = get(hafter,'Parent');
assert(get(hnew,'Parent') == hpar);
newpos = get(hafter,'Position');
set(hnew,'Position',newpos);
