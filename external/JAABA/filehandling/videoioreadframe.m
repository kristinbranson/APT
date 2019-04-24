function im = videoioreadframe(readerobj,f)

seek(readerobj,double(f));
im = getframe(readerobj);