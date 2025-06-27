function moveMenuItemAfter(hnew,hprev)

warning('Obsolete. Position argument does not seem to reliably work any more.');

hpar = get(hprev,'Parent');
assert(get(hnew,'Parent') == hpar);
newpos = min(get(hprev,'Position') + 1,numel(get(hpar,'Children')));
set(hnew,'Position',newpos);
