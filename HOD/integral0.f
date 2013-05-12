      program integral

      implicit none

      integer i,lines
      double precision r(1000),chi(1000)

      integer nvar,nok,nbad,ep,npar
      double precision h1,hmin,eps,PAR(10000),ystart(1),r1,r2,x
      double precision pi,chi_aux,interp,rp(13)
      external derivs,rkqs,escribir
      
      common lines,r,chi

      rp(1)=0.17
      rp(2)=0.27
      rp(3)=0.42
      rp(4)=0.67
      rp(5)=1.1
      rp(6)=1.7
      rp(7)=2.7
      rp(8)=4.2
      rp(9)=6.7
      rp(10)=10.6
      rp(11)=16.9
      rp(12)=26.8
      rp(13)=42.3


      pi=dacos(-1d0)

c if number of lines larger than 1000 change numbers in the variable definition
c and in the interp function
c######## PARAMETERS #########
c     name of of file and number of lines
      open(unit=10,file='borrar20.dat',status='unknown')
      lines=29
c#############################
      
      do i=1,lines
         read(10,*) r(i),chi(i)
      end do
      close(unit=10)

      npar=1

      nvar=1
      h1=1d-10
      hmin=0d0
      eps=1d-9

      ep=1

c      open (unit=10,file='borrar3.dat',status='unknown')
c      do i=1,10000
c         x=r(1)+(r(lines)-r(1))*i/10000d0
c         chi_aux=interp(x)
c         write(10,*) x,chi_aux
c      end do

      open (unit=10,file='borrar30.dat',status='unknown')
      do i=1,13
         x=rp(i)
         PAR(1)=x

         r1=rp(i)+1d-8
         r2=r(lines)

         ystart(1)=0d0

         call odeint(ystart,nvar,r1,r2,eps,h1,hmin,nok,nbad,derivs,
     &        rkqs,npar,PAR,ep,escribir)

         write (*,*) x,ystart(1)

         write(10,*) x,ystart(1)
      end do

      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      double precision function interp (rp)
      
      integer i,lines
      common lines

      double precision rp,r(1000),chi(1000)
      common r,chi

      if (rp<=r(1)) then
         interp=chi(1)
      else if (rp>=r(lines)) then
         interp=chi(lines)
      else
         i=1
         do while (rp>r(i))
            i=i+1           
         end do
         interp=(chi(i)-chi(i-1))/(r(i)-r(i-1))*(rp-r(i-1))+chi(i-1)
      end if

      return
      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine derivs(x,y,dydx,npar,PAR)

      implicit none

      integer npar
      double precision x,y(2),dydx(2),PAR(npar)
      
      double precision rp,interp,chi

      rp=PAR(1)

      chi=interp(x)

      dydx(1)=2d0*x*chi/sqrt(x**2-rp**2)

      return
      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine escribir (x,y,npar,PAR)

      implicit none

      integer npar
      double precision x,y(2),PAR(npar)

c      write (1,*) x,y(1)

      return
      end
