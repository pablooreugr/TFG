import numpy as np

def write_Noll(j):
    k=0
    n=-1
    jlist=np.arange(1,j+1)
    mlist=[]
    nlist=[]
    while k<j:
        n+=1
        mvec=np.arange(-n,n+1)
        mvec_bool=np.where((n+mvec)%2==0,True,False) #n+m must be even
        mvec=np.extract(mvec_bool,mvec)
        absmvec=np.abs(mvec)
        indices=np.argsort(absmvec)
        mvec=mvec[indices]
        mvec2=np.copy(mvec)
        for m in mvec:
            k+=1
            if k<=j:
                if k%2!=0:
                    if m<0:
                        mzj=m
                        mlist.append(mzj)
                        nlist.append(n)
                    else:
                        mzj=-m
                        mlist.append(mzj)
                        nlist.append(n)
                else:
                    if m>=0:
                        mzj=m
                        mlist.append(mzj)
                        nlist.append(n)
                    else:
                        mzj=-m
                        mlist.append(mzj)
                        nlist.append(n)
                mvec2=np.delete(mvec2,np.where(mvec2==m))
            else:break
    return jlist,mlist,nlist

jlist, mlist, nlist = write_Noll(200)
# zernikej_Noll(j,rho,theta) uses:
# m=zernike_equiv[j-1,1]
# n=zernike_equiv[j-1,2]
# So column 0 is j, column 1 is m, column 2 is n
np.savetxt('j_to_Noll.txt', np.column_stack((jlist, mlist, nlist)), fmt='%d')
print("Generated j_to_Noll.txt")
