function moveMenuItemAfter(hnew,hprev)

hpar = get(hprev,'Parent');
assert(get(hnew,'Parent') == hpar);
newpos = get(hprev,'Position') + 1;
set(hnew,'Position',newpos);
