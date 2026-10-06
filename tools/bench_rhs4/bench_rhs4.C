// Standalone check and timing of the Cartesian RHS kernels (src/rhs4th3fortc.C).
//
//   bench_rhs4 check > out.bin    '=' then '-' (as EW::evalRHS with attenuation) on random data,
//                                 both kernels, every onesided combination; lu written raw
//   bench_rhs4 time [n ...]       ms per rhs4th3fortsgstr_ci call (OP '='), n^3 points with
//                                 2 ghost points, free surface on top (default: 64 305)
//
// Link with the rhs4th3fortc.C to test; compare two builds' check outputs with compare.py.
#include "sw4.h"
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void rhs4th3fort_ci( int, int, int, int, int, int, int, int*, float_sw4*, float_sw4*, float_sw4*,
                     float_sw4*, float_sw4*, float_sw4*, float_sw4*, float_sw4, char );
void rhs4th3fortsgstr_ci( int, int, int, int, int, int, int, int*, float_sw4*, float_sw4*, float_sw4*,
                          float_sw4*, float_sw4*, float_sw4*, float_sw4*, float_sw4, float_sw4*, float_sw4*,
                          float_sw4*, char );

static void fill( std::vector<float_sw4>& v, unsigned& s, float_sw4 lo, float_sw4 width )
{
   for( auto& x : v ) { s = s*1103515245u+12345u; x = lo + width*((s>>8)*(1.0f/16777216.0f)); }
}

static int check()
{
   const int ifirst=-1, ilast=14, jfirst=-1, jlast=13, kfirst=-1, nk=17, klast=nk+2;
   const size_t npts = (size_t)(ilast-ifirst+1)*(jlast-jfirst+1)*(klast-kfirst+1);
   unsigned s = 12345;
   std::vector<float_sw4> u(3*npts), al(3*npts), mu(npts), la(npts), mua(npts), laa(npts), lu(3*npts);
   std::vector<float_sw4> acof(384), bope(48), ghcof(6), sx(ilast-ifirst+1), sy(jlast-jfirst+1), sz(klast-kfirst+1);
   fill(u,s,-1,2); fill(al,s,-1,2); fill(mu,s,1,1); fill(la,s,1,1); fill(mua,s,1,1); fill(laa,s,1,1);
   fill(acof,s,-1,2); fill(bope,s,-1,2); fill(ghcof,s,-1,2); fill(sx,s,0.5,1); fill(sy,s,0.5,1); fill(sz,s,0.5,1);
   for( int kern=0 ; kern < 2 ; kern++ )
      for( int os=0 ; os < 4 ; os++ )
      {
         int onesided[6] = {0,0,0,0, os&1, (os>>1)&1};
         for( auto& x : lu ) x = 7.0f;
         for( int pass=0 ; pass < 2 ; pass++ )
         {
            float_sw4* in = pass == 0 ? u.data() : al.data();
            float_sw4* m  = pass == 0 ? mu.data() : mua.data();
            float_sw4* l  = pass == 0 ? la.data() : laa.data();
            char op = pass == 0 ? '=' : '-';
            if( kern == 0 )
               rhs4th3fort_ci(ifirst,ilast,jfirst,jlast,kfirst,klast,nk,onesided,acof.data(),bope.data(),
                              ghcof.data(),lu.data(),in,m,l,0.1f,op);
            else
               rhs4th3fortsgstr_ci(ifirst,ilast,jfirst,jlast,kfirst,klast,nk,onesided,acof.data(),bope.data(),
                                   ghcof.data(),lu.data(),in,m,l,0.1f,sx.data(),sy.data(),sz.data(),op);
         }
         fwrite(lu.data(), sizeof(float_sw4), lu.size(), stdout);
      }
   return 0;
}

static int timing( std::vector<int> sizes )
{
   if( sizes.empty() )
      sizes = {64, 305};
   const char* rank = getenv("SLURM_PROCID");
   printf("rank %s, %d OpenMP threads\n", rank ? rank : "-", omp_get_max_threads());
   printf("%6s %12s %10s %10s %12s\n", "n", "Mpoints", "min (ms)", "mean (ms)", "ns/point");
   for( int n : sizes )
   {
      // n points per direction including 2 ghost points on each side, as an Sarray
      const int ifirst=-1, ilast=n-2, jfirst=-1, jlast=n-2, kfirst=-1, klast=n-2, nk=n-4;
      const size_t npts = (size_t)n*n*n;
      unsigned s = 4321;
      std::vector<float_sw4> u(3*npts), mu(npts), la(npts), lu(3*npts);
      std::vector<float_sw4> acof(384), bope(48), ghcof(6), sx(n), sy(n), sz(n);
      // first touch in parallel, by k-planes, as the solver's arrays
#pragma omp parallel for
      for( int k=0 ; k < n ; k++ )
      {
         size_t o = (size_t)k*n*n;
         for( size_t q=0 ; q < (size_t)n*n ; q++ )
         {
            mu[o+q] = la[o+q] = 1;
            for( int c=0 ; c < 3 ; c++ ) { u[c*npts+o+q] = 0; lu[c*npts+o+q] = 0; }
         }
      }
      fill(u,s,-1,2); fill(mu,s,1,1); fill(la,s,1,1);
      fill(acof,s,-1,2); fill(bope,s,-1,2); fill(ghcof,s,-1,2); fill(sx,s,0.5,1); fill(sy,s,0.5,1); fill(sz,s,0.5,1);
      int onesided[6] = {0,0,0,0,1,0};
      const int reps = std::max(5, (int)(3e8/npts));
      double tmin = 1e30, tsum = 0;
      for( int r=-1 ; r < reps ; r++ )
      {
         double t0 = omp_get_wtime();
         rhs4th3fortsgstr_ci(ifirst,ilast,jfirst,jlast,kfirst,klast,nk,onesided,acof.data(),bope.data(),
                             ghcof.data(),lu.data(),u.data(),mu.data(),la.data(),0.1f,sx.data(),sy.data(),
                             sz.data(),'=');
         double t = omp_get_wtime() - t0;
         if( r < 0 ) continue;   // warm-up
         tmin = std::min(tmin, t); tsum += t;
      }
      const double interior = (double)(n-4)*(n-4)*(n-4);
      printf("%6d %12.2f %10.3f %10.3f %12.3f\n", n, interior/1e6, 1e3*tmin, 1e3*tsum/reps, 1e9*tmin/interior);
      fflush(stdout);
   }
   return 0;
}

int main( int argc, char** argv )
{
   if( argc > 1 && !strcmp(argv[1], "check") )
      return check();
   std::vector<int> sizes;
   for( int a=2 ; a < argc ; a++ )
      sizes.push_back(atoi(argv[a]));
   return timing(sizes);
}
