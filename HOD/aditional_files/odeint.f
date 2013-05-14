      subroutine odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,
     &     rkqs,npar,PAR,ep,escribir)

      implicit none

      integer nbad,nok,nvar,KMAXX,MAXSTP,NMAX,npar,ep
      double precision eps,h1,hmin,x1,x2,ystart(nvar),TINY
      double precision PAR(npar)
      external derivs,rkqs,escribir
      parameter (MAXSTP=1000000,NMAX=50,KMAXX=200,TINY=1D-30)
      integer i,kmax,kount,nstp
      double precision dxsav,h,hdid,hnext,x,xsav,dydx(NMAX),xp(KMAXX),
     &                       y(NMAX),yp(NMAX,KMAXX),yscal(NMAX)
      common /path/ kmax,kount,dxsav,xp,yp

      x=x1
      h=sign(h1,x2-x1)
      nok=0
      nbad=0
      kount=0
      do i=1,nvar
         y(i)=ystart(i)
      enddo
      if (kmax.gt.0) xsav=x-2d0*dxsav
      do nstp=1,MAXSTP
         call derivs(x,y,dydx,npar,PAR)
         do i=1,nvar
            yscal(i)=abs(y(i))+abs(h*dydx(i))+TINY
         enddo
         if (kmax.gt.0) then
            if(abs(x-xsav).gt.abs(dxsav)) then
               if(kount.lt.kmax-1) then
                  kount=kount+1
                  xp(kount)=x
                  do i=1,nvar
                     yp(i,kount)=y(i)
                  enddo
                  xsav=x
               end if
            end if
         end if
         if((x+h-x2)*(x+h-x1).gt.0d0) h=x2-x
         call rkqs(y,dydx,nvar,x,h,eps,yscal,hdid,hnext,derivs,npar,PAR)
         if(hdid.eq.h) then
            if (ep==1) then
               call escribir (x,y,npar,PAR)
            end if
            nok=nok+1
         else
            nbad=nbad+1
         end if
         if((x-x2)*(x2-x1).ge.0) then
            do i=1,nvar
               ystart(i)=y(i)
            enddo
            if(kmax.ne.0) then
               kount=kount+1
               xp(kount)=x
               do i=1,nvar
                  yp(i,kount)=y(i)
               enddo
            end if
            return
         end if
         if (abs(hnext).lt.hmin) stop 'stepsize smaller than ninimum'
         h=hnext
      enddo
      stop 'too many steps in odeint'
      return
      end


      subroutine rkqs(y,dydx,n,x,htry,eps,yscal,hdid,hnext,derivs,npar,
     &     PAR)
     
      implicit none

      integer n,NMAX,npar
      double precision eps,hdid,hnext,htry,x,dydx(n),y(n),yscal(n)
      double precision PAR(npar)
      external derivs
      parameter (NMAX=50)
      integer i
      double precision errmax,h,htemp,xnew,yerr(NMAX),ytemp(NMAX),
     &          SAFETY,PGROW,PSHRNK,ERRCON
      PARAMETER (SAFETY=0.9,PGROW=-0.2,PSHRNK=-0.25,ERRCON=1.89d-4)
      h=htry
 1    call rkck(y,dydx,n,x,h,ytemp,yerr,derivs,npar,PAR)
      errmax=0
      do i=1,n
         errmax=max(errmax,abs(yerr(i)/yscal(i)))
      enddo
      errmax=errmax/eps
      if(errmax.gt.1.0) then
         htemp=SAFETY*h*(errmax**PSHRNK)
         h=sign(max(abs(htemp),0.1*abs(h)),h)
         xnew=x+h
         if(xnew.eq.x) stop 'stepsize underflow in rkqs'
         goto 1
      else
         if(errmax.gt.ERRCON) then
            hnext=SAFETY*h*(errmax**PGROW)
         else
            hnext=5.0*h
         end if
         hdid=h
         x=x+h
         do i=1,n
            y(i)=ytemp(i)
         enddo
         return
      endif
      end


      subroutine rkck(y,dydx,n,x,h,yout,yerr,derivs,npar,PAR)

      implicit none
      
      integer n,NMAX,npar
      double precision h,x,dydx(n),yerr(n),yout(n),y(n)
      double precision PAR(npar)
      external derivs
      parameter (NMAX=50) 
      integer i
      double precision ak2(NMAX),ak3(NMAX),ak4(NMAX),ak5(NMAX),
     &                      ak6(NMAX),ytemp(NMAX),A2,A3,A4,A5,A6,B21,
     &                      B31,B32,B41,B42,B43,B51,B52,B53,B54,B61,
     &                      B62,B63,B64,B65,C1,C3,C4,C6,DC1,DC3,DC4,
     &                      DC5,DC6
      parameter (A2=0.2,A3=0.3,A4=0.6,A5=1.0,A6=0.875,B21=0.2,
     &             B31=3.0/40.0,B32=9.0/40.0,B41=0.3,B42=-0.9,
     &             B43=1.2,B51=-11.0/54.0,B52=2.5,B53=-70.0/27.0,
     &             B54=35.0/27.0,B61=1631.0/55296.0,B62=175.0/512.0,
     &             B63=575.0/13824.0,B64=44275.0/110592.0,
     &             B65=253.0/4096.0,C1=37.0/378.0,C3=250.0/621.0,
     &             C4=125.0/594.0,C6=512.0/1771.0,DC1=C1-2825.0/27648.0,
     &             DC3=C3-18575.0/48384.0,DC4=C4-13525.0/55296.0, 
     &             DC5=-277.0/14336.0,DC6=C6-0.25)
      do i=1,n
         ytemp(i)=y(i)+B21*h*dydx(i)
      enddo
      call derivs(x+A2*h,ytemp,ak2,npar,PAR)
      do i=1,n
         ytemp(i)=y(i)+h*(B31*dydx(i)+B32*ak2(i))
      enddo
      call derivs(x+A3*h,ytemp,ak3,npar,PAR)
      do i=1,n
         ytemp(i)=y(i)+h*(B41*dydx(i)+B42*ak2(i)+B43*ak3(i))
      enddo
      call derivs(x+A4*h,ytemp,ak4,npar,PAR)
      do i=1,n
         ytemp(i)=y(i)+h*(B51*dydx(i)+B52*ak2(i)+B53*ak3(i)+B54*ak4(i))
      enddo
      call derivs(x+A5*h,ytemp,ak5,npar,PAR)
      do i=1,n
         ytemp(i)=y(i)+h*(B61*dydx(i)+B62*ak2(i)+B63*ak3(i)+B64*ak4(i)+
     &        B65*ak5(i))
      enddo
      call derivs(x+A6*h,ytemp,ak6,npar,PAR)
      do i=1,n
         yout(i)=y(i)+h*(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i))
      enddo
      do i=1,n
         yerr(i)=h*(DC1*dydx(i)+DC3*ak3(i)+DC4*ak4(i)+DC5*ak5(i)+
     &        DC6*ak6(i))
      enddo
      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
