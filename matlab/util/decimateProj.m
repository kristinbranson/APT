function decimateProj(lObj,nkeepApprox)

% Warning: this method doesn't work for MA projs yet,
% bc the labelGetMFTableLabeled call doesn't return all tgts.

t = lObj.labelGetMFTableLabeled;
n = height(t);
nRm = n-nkeepApprox;
rmfac = nRm/n;
fprintf(1,'Removing %.2f frac of labels\n',rmfac);

for i=1:n
  if rand < rmfac
    s = t(i,:);
    lObj.setMFTGUI(s.mov,s.frm,s.iTgt);
    if lObj.maIsMA
      lObj.labelPosClearWithCompact_New();
    else
      lObj.labelPosClear();
    end
    fprintf(1,'Cleared (%d,%d,%d)\n',s.mov,s.frm,s.iTgt);
  end
end

fprintf(1,'Done!\n');