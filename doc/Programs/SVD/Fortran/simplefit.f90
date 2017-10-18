!    Program to fit a given set of data in terms of a given polynomial using 
!    Singular Value Decomposition, see Golub and Van Loan, Matrix Computations
!    Numerical Recipes, chapter 15.
!    This explict example fits a polynomial to a given degree based. The example
!    input should look like
!     8 2    #  number of data entries and order of polynomial
!     0.001 	-2.89017 	0.00073621
!     0.002 	-2.88946 	0.00052732
!     0.005	-2.89067  	0.00055038
!     0.010 	-2.89091 	0.00040973
!     0.015 	-2.89084 	0.00034278
!     0.02  	-2.89086 	0.00029315
!     0.025 	-2.89059	0.00034278
!     0.03 	-2.89077 	0.00025017
!    The present input file  stems from a variational Monte Carlo calculation of the ground
!    state energy (second column). It contains also the standard deviation (3rd column)
!    The first column is the time step used in importance sampling. Extrapolating to zero 
!    should give the best estimate for the VMC energy.
!
!    Author: Morten Hjorth-Jensen, Department of Physics, University of Oslo,
!    POB 1048 Blindern, N-0316 Oslo, Norway. email: morten.hjorth-jensen@fys.uio.no 
!    Latest update: October 1999.
!    Run in old fashioned way, executable < inputfile  > outputfile

!
!     Main program starts here
!
      PROGRAM  fitting
      IMPLICIT NONE
      CALL vmcfit

      END  PROGRAM fitting
!
!     This subroutine fits a data set with  in terms
!     of a polynomial expansion. The number of terms in the polynomial
!     expansion is given by the variable number_terms
!     The number of data ( delta  and corresponding energy
!     and the standard error) are defined by the variable number_of_data
!
      SUBROUTINE  vmcfit
      IMPLICIT NONE
      INTEGER :: i, number_of_data, number_terms
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: w, n, e,sig,polynom_terms, variance
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: cvm, v, u
      DOUBLE PRECISION :: chisq, kappa, v_over_c, adiabatic
      EXTERNAL eos_param

      READ (5,*) number_of_data, number_terms                  
      ALLOCATE (cvm(number_terms,number_terms), &
                v(number_terms,number_terms) )
      ALLOCATE ( u(number_of_data,number_terms) )
      ALLOCATE ( w(number_terms) )
      ALLOCATE ( variance(number_terms) )
      ALLOCATE ( polynom_terms(number_terms) )
      ALLOCATE (n(number_of_data), e(number_of_data), sig(number_of_data) ) 
      DO i=1,number_of_data
         READ (5,*)n(i) , e(i), sig(i)
      ENDDO
      CALL svdfit(n,e,sig,number_of_data,polynom_terms, &
                  number_terms, &
                  u,v,w,number_of_data,number_terms,chisq)
      CALL svdvar(v,number_terms,number_terms,w,cvm,number_terms)
!     Compute now the variance for each coefficient using Eq. 15.4.19
!     of numerical recipes
      variance = 0.0; w = 1.0/(w*w)
      DO i = 1, number_terms
         variance(i) = SUM(v(i,:)*v(i,:)*w(:))
      ENDDO
      WRITE(*,*) ' Quality of fit:'
      WRITE(*,*) ' Number of terms in polynomial expansion:',  number_terms
      WRITE (*,'(''CHISQ = '',F12.5)') chisq
      WRITE(*,*) 'Extrapolated energy with error'
      WRITE(*,*) polynom_terms(1), SQRT(variance(1))
      DEALLOCATE (cvm, v)
      DEALLOCATE ( u )
      DEALLOCATE ( w,polynom_terms, variance )
      DEALLOCATE (n, e, sig) 

      END SUBROUTINE vmcfit


!     This function encodes the actual functional form of the polynomial.
!     If you need to change the functional form, this is the function to adjust.
!     This specific version is a polynomial in x^i, with i >=0 and being an integer

      SUBROUTINE eos_param(x,afunc,ma)
      IMPLICIT NONE
      INTEGER :: ma, i
      DOUBLE PRECISION :: afunc, x
      DIMENSION afunc(ma)
      afunc(1)=1.D0
      DO i=2,ma
         afunc(i)=afunc(i-1)*x
      ENDDO

      END SUBROUTINE eos_param



      SUBROUTINE svdvar(v,ma,np,w,cvm,ncvm)
      IMPLICIT NONE
      INTEGER :: ma,ncvm,np,MMAX
      DOUBLE PRECISION ::  cvm(ncvm,ncvm),v(np,np),w(np)
      PARAMETER (MMAX=100)
      INTEGER :: i,j,k
      DOUBLE PRECISION :: sum,wti(MMAX)
      DO i=1,ma
        wti(i)=0.
        IF (w(i) /= 0.) wti(i)=1./(w(i)*w(i))
      ENDDO
      DO i=1,ma
         DO j=1,i
            sum=0.
            DO k=1,ma
               sum=sum+v(i,k)*v(j,k)*wti(k)
            ENDDO
            cvm(i,j)=sum
            cvm(j,i)=sum
         ENDDO
      ENDDO

      END SUBROUTINE svdvar


      SUBROUTINE svbksb(u,w,v,m,n,mp,np,b,x)
      IMPLICIT NONE
      INTEGER :: m,mp,n,np,NMAX
      DOUBLE PRECISION :: b(mp),u(mp,np),v(np,np),w(np),x(np)
      PARAMETER (NMAX=5000)
      INTEGER :: i,j,jj
      DOUBLE PRECISION :: s,tmp(NMAX)
      DO j=1,n
         s=0.
         IF (w(j) /= 0.) THEN
            DO i=1,m
               s=s+u(i,j)*b(i)
            ENDDO
            s=s/w(j)
         ENDIF
         tmp(j)=s
      ENDDO
      DO j=1,n
         s=0.
         DO jj=1,n
            s=s+v(j,jj)*tmp(jj)
         ENDDO
         x(j)=s
      ENDDO

      END SUBROUTINE svbksb


      SUBROUTINE svdfit(x,y,sig,ndata,a,ma,u,v,w,mp,np,chisq)
      IMPLICIT NONE
      INTEGER :: ma,mp,ndata,np,NMAX,MMAX
      DOUBLE PRECISION :: chisq,a(ma),sig(ndata),u(mp,np),v(np,np),w(np), &
                          x(ndata),  y(ndata),TOL
      PARAMETER (NMAX=5000,MMAX=100,TOL=1.e-5)
      INTEGER :: i,j
      DOUBLE PRECISION :: sum,thresh,tmp,wmax,afunc(MMAX), &
                          b(NMAX)

      DO i=1,ndata
         CALL  eos_param(x(i),afunc,ma)
         tmp=1./sig(i)
         DO j=1,ma
            u(i,j)=afunc(j)*tmp
         ENDDO
         b(i)=y(i)*tmp
      ENDDO
      CALL svdcmp(u,ndata,ma,mp,np,w,v)
      wmax=0.
      DO j=1,ma
         IF (w(j) > wmax)wmax=w(j)
      ENDDO
      thresh=TOL*wmax
      DO j=1,ma
         IF (w(j) < thresh)  w(j)=0.
      ENDDO
      CALL svbksb(u,w,v,ndata,ma,mp,np,b,a)
      chisq=0.
      DO i=1,ndata
         CALL eos_param(x(i),afunc,ma)
         sum=0.
         DO j=1,ma
            sum=sum+a(j)*afunc(j)
         ENDDO
         chisq=chisq+((y(i)-sum)/sig(i))**2
      ENDDO

      END SUBROUTINE svdfit


      FUNCTION pythag(a,b)
      IMPLICIT NONE
      DOUBLE PRECISION ::  a, b, pythag 
      DOUBLE PRECISION :: absa, absb
      absa=ABS(a)
      absb=ABS(b)
      IF (absa > absb) THEN
         pythag=absa*SQRT(1.+(absb/absa)**2)
      ELSE
         IF(absb == 0.) THEN
            pythag=0.
         ELSE
            pythag=absb*SQRT(1.+(absa/absb)**2)
         ENDIF
      ENDIF

      END FUNCTION pythag

      SUBROUTINE svdcmp(a,m,n,mp,np,w,v)
      IMPLICIT NONE
      INTEGER :: m,mp,n,np,NMAX
      DOUBLE PRECISION :: a(mp,np),v(np,np),w(np)
      PARAMETER (NMAX=5000)
      INTEGER :: i,its,j,jj,k,l,nm
      DOUBLE PRECISION :: anorm,c,f,g,h,s,scale,x,y,z, &
                          rv1(NMAX),pythag
      g=0.0
      scale=0.0
      anorm=0.0
      DO i=1,n
         l=i+1
         rv1(i)=scale*g
         g=0.0
         s=0.0
         scale=0.0
         IF(i <= m)THEN
            DO k=i,m
               scale=scale+ABS(a(k,i))
            ENDDO
            IF(scale /= 0.0)THEN
               DO k=i,m
                  a(k,i)=a(k,i)/scale
                  s=s+a(k,i)*a(k,i)
               ENDDO
               f=a(i,i)
               g=-sign(SQRT(s),f)
               h=f*g-s
               a(i,i)=f-g
               DO j=l,n
                  s=0.0
                  DO k=i,m
                     s=s+a(k,i)*a(k,j)
                  ENDDO
                  f=s/h
                  DO k=i,m
                     a(k,j)=a(k,j)+f*a(k,i)
                  ENDDO
               ENDDO
               DO k=i,m
                  a(k,i)=scale*a(k,i)
               ENDDO
            ENDIF
         ENDIF
         w(i)=scale *g
         g=0.0
         s=0.0
         scale=0.0
         IF((i <= m).and.(i /= n))THEN
            DO k=l,n
               scale=scale+ABS(a(i,k))
            ENDDO
            IF(scale /= 0.0)THEN
               DO k=l,n
                  a(i,k)=a(i,k)/scale
                  s=s+a(i,k)*a(i,k)
               ENDDO
               f=a(i,l)
               g=-sign(SQRT(s),f)
               h=f*g-s
               a(i,l)=f-g
               DO k=l,n
                  rv1(k)=a(i,k)/h
               ENDDO
               DO j=l,m
                  s=0.0
                  DO k=l,n
                     s=s+a(j,k)*a(i,k)
                  ENDDO
                  DO k=l,n
                     a(j,k)=a(j,k)+s*rv1(k)
                  ENDDO
               ENDDO
               DO k=l,n
                  a(i,k)=scale*a(i,k)
               ENDDO
            ENDIF
         ENDIF
         anorm=max(anorm,(ABS(w(i))+ABS(rv1(i))))
      ENDDO
      DO i=n,1,-1
         IF(i < n)THEN
            IF(g /= 0.0)THEN
               DO j=l,n
                  v(j,i)=(a(i,j)/a(i,l))/g
               ENDDO
               DO j=l,n
                  s=0.0
                  DO k=l,n
                     s=s+a(i,k)*v(k,j)
                  ENDDO
                  DO k=l,n
                     v(k,j)=v(k,j)+s*v(k,i)
                  ENDDO
               ENDDO
            ENDIF
            DO j=l,n
               v(i,j)=0.0
               v(j,i)=0.0
            ENDDO
         ENDIF
         v(i,i)=1.0
         g=rv1(i)
         l=i
      ENDDO
      DO i=min(m,n),1,-1
         l=i+1
         g=w(i)
         DO j=l,n
            a(i,j)=0.0
         ENDDO
         IF(g /= 0.0)THEN
            g=1.0/g
            DO j=l,n
               s=0.0
               DO k=l,m
                  s=s+a(k,i)*a(k,j)
               ENDDO
               f=(s/a(i,i))*g
               DO k=i,m
                  a(k,j)=a(k,j)+f*a(k,i)
               ENDDO
            ENDDO
            DO j=i,m
               a(j,i)=a(j,i)*g
            ENDDO
         ELSE
            DO j= i,m
               a(j,i)=0.0
            ENDDO
         ENDIF
         a(i,i)=a(i,i)+1.0
      ENDDO
      DO k=n,1,-1
         DO its=1,30
            DO l=k,1,-1
               nm=l-1
               IF((ABS(rv1(l))+anorm) == anorm)  goto 2
               IF((ABS(w(nm))+anorm) == anorm)  goto 1
            ENDDO
1           c=0.0
            s=1.0
            DO i=l,k
               f=s*rv1(i)
               rv1(i)=c*rv1(i)
               IF((ABS(f)+anorm) == anorm) goto 2
               g=w(i)
               h=pythag(f,g)
               w(i)=h
               h=1.0/h
               c= (g*h)
               s=-(f*h)
               DO j=1,m
                  y=a(j,nm)
                  z=a(j,i)
                  a(j,nm)=(y*c)+(z*s)
                  a(j,i)=-(y*s)+(z*c)
               ENDDO
            ENDDO
2           z=w(k)
            IF(l == k)THEN
               IF(z < 0.0)THEN
                  w(k)=-z
                  DO j=1,n
                     v(j,k)=-v(j,k)
                  ENDDO
               ENDIF
               goto 3
            ENDIF
            IF(its == 30) THEN 
               WRITE(*,*) 'no convergence in svdcmp'; STOP
            ENDIF
            x=w(l)
            nm=k-1
            y=w(nm)
            g=rv1(nm)
            h=rv1(k)
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g=pythag(f,1.D0)
            f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
            c=1.0
            s=1.0
            DO j=l,nm
               i=j+1
               g=rv1(i)
               y=w(i)
               h=s*g
               g=c*g
               z=pythag(f,h)
               rv1(j)=z
               c=f/z
               s=h/z
               f= (x*c)+(g*s)
               g=-(x*s)+(g*c)
               h=y*s
               y=y*c
               DO jj=1,n
                  x=v(jj,j)
                  z=v(jj,i)
                  v(jj,j)= (x*c)+(z*s)
                  v(jj,i)=-(x*s)+(z*c)
               ENDDO
               z=pythag(f,h)
               w(j)=z
               IF(z /= 0.0)THEN
                  z=1.0/z
                  c=f*z
                  s=h*z
               ENDIF
               f= (c*g)+(s*y)
               x=-(s*g)+(c*y)
               DO jj=1,m
                  y=a(jj,j)
                  z=a(jj,i)
                  a(jj,j)= (y*c)+(z*s)
                  a(jj,i)=-(y*s)+(z*c)
               ENDDO
            ENDDO
            rv1(l)=0.0
            rv1(k)=f
            w(k)=x
         ENDDO
3        CONTINUE    
      ENDDO

      END SUBROUTINE svdcmp






