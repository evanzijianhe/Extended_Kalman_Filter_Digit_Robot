function Qk = Qk_val
    Qrx = 0.1;
    Qry = 0.1;
    Qrz = 0.1;
    Qr = blkdiag(Qrx, Qry, Qrz);
        
    Qvx = 0.1;
    Qvy = 0.1;
    Qvz = 0.1;
    Qv = blkdiag(Qvx, Qvy, Qvz);
    
    Qqw = 0.01;
    Qqx = 0.01;
    Qqy = 0.01;
    Qqz = 0.01;
    Qq = blkdiag(Qqw,Qqx, Qqy, Qqz);
    
    Qfoot = 0.01*eye(6);
    Qk = blkdiag(Qr, Qv, Qq,Qfoot); 
    
end