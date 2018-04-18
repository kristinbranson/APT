function [x,y,th,a,b] = readtrx(trx,frm,iTgt)
trxI = trx(iTgt);
idx = trxI.off+frm;
x = trxI.x(idx);
y = trxI.y(idx);
th = trxI.theta(idx);
a = trxI.a(idx);
b = trxI.b(idx);