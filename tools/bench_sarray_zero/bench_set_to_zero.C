// Standalone benchmark of Sarray::set_to_zero (src/Sarray.C), which EW::Force and EW::Force_tt
// call on the whole forcing array every time step.
//
//   bench_set_to_zero [n ...]      cubes of n^3 points with 3 components (default: 32 64 100 150 200 305)
//
// The array is float (a USE_DOUBLE=OFF build) and is first touched with the current loop, as
// solve.C does at startup. Every variant is timed on that array and checked to leave only zeros.
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef float float_sw4;

struct Arr {
   int ni, nj, nk, nc;
   size_t npts;
   float_sw4* data;
};

// the current loop: one parallel region per component
static void zero_orig( Arr& a )
{
   const int ni = a.ni, nj = a.nj, nk = a.nk;
   const size_t offj = ni, offk = (size_t)ni*nj, offc = a.npts;
   float_sw4* m_data = a.data;
   for( int c=0 ; c < a.nc ; c++ )
#pragma omp parallel for collapse(2)
      for( int k=0 ; k < nk ; k++ )
         for( int j=0 ; j < nj ; j++ )
#pragma omp simd
            for( int i=0 ; i < ni ; i++ )
               m_data[offc*c+i+offj*j+offk*k] = 0;
}

// one parallel region; static schedule with the same bounds -> same (k,j) chunk per thread for
// every c, so the thread->page mapping of the first touch is kept
static void zero_region( Arr& a )
{
   const int ni = a.ni, nj = a.nj, nk = a.nk, nc = a.nc;
   const size_t offj = ni, offk = (size_t)ni*nj, offc = a.npts;
   float_sw4* m_data = a.data;
#pragma omp parallel
   for( int c=0 ; c < nc ; c++ )
   {
#pragma omp for collapse(2) schedule(static) nowait
      for( int k=0 ; k < nk ; k++ )
         for( int j=0 ; j < nj ; j++ )
#pragma omp simd
            for( int i=0 ; i < ni ; i++ )
               m_data[offc*c+i+offj*j+offk*k] = 0;
   }
}

// as zero_region, with streaming (non-temporal) stores where the compiler has a pragma for them
static void zero_region_nt( Arr& a )
{
   const int ni = a.ni, nj = a.nj, nk = a.nk, nc = a.nc;
   const size_t offj = ni, offk = (size_t)ni*nj, offc = a.npts;
   float_sw4* m_data = a.data;
#pragma omp parallel
   for( int c=0 ; c < nc ; c++ )
   {
#pragma omp for collapse(2) schedule(static) nowait
      for( int k=0 ; k < nk ; k++ )
         for( int j=0 ; j < nj ; j++ )
#ifdef __INTEL_COMPILER
#pragma vector nontemporal
#endif
#pragma omp simd
            for( int i=0 ; i < ni ; i++ )
               m_data[offc*c+i+offj*j+offk*k] = 0;
   }
}

// one parallel region; each thread memsets its static share of the (k,j) lines of each
// component, which is one contiguous block because the loops cover the whole array.
// The split is the one libgomp and the LLVM/Intel runtime use for schedule(static):
// the first n%nt threads get one line more.
static void zero_memset( Arr& a )
{
   const size_t nlines = (size_t)a.nj*a.nk, line = a.ni;
   float_sw4* m_data = a.data;
   const int nc = a.nc;
   const size_t offc = a.npts;
#pragma omp parallel
   {
      const size_t nt = omp_get_num_threads(), t = omp_get_thread_num();
      const size_t q = nlines/nt, r = nlines%nt;
      const size_t first = t*q + std::min(t, r);
      const size_t count = q + (t < r ? 1 : 0);
      for( int c=0 ; c < nc ; c++ )
         memset(m_data + offc*c + first*line, 0, count*line*sizeof(float_sw4));
   }
}

typedef void (*zero_fn)( Arr& );

int main( int argc, char** argv )
{
   std::vector<int> sizes;
   for( int a=1 ; a < argc ; a++ )
      sizes.push_back(atoi(argv[a]));
   if( sizes.empty() )
      sizes = {32, 64, 100, 150, 200, 305};

   const char* names[] = {"orig", "region", "region-nt", "memset"};
   zero_fn fns[] = {zero_orig, zero_region, zero_region_nt, zero_memset};
   const int nvar = 4;
   const char* rank = getenv("SLURM_PROCID");

#ifdef __INTEL_COMPILER
   const char* nt_note = "";
#else
   const char* nt_note = " (region-nt = region: no nontemporal pragma for this compiler)";
#endif
   printf("rank %s, %d OpenMP threads%s\n", rank ? rank : "-", omp_get_max_threads(), nt_note);
   printf("%6s %10s %-10s %10s %10s %9s %s\n", "n", "MB", "variant", "min (ms)", "mean (ms)", "GB/s", "check");
   for( int n : sizes )
   {
      Arr a;
      a.ni = a.nj = a.nk = n;
      a.nc = 3;
      a.npts = (size_t)n*n*n;
      const size_t nval = a.npts*a.nc;
      a.data = new float_sw4[nval];
      zero_orig(a);   // first touch, as solve.C

      const int reps = std::max(5, (int)(4e9/(nval*sizeof(float_sw4))));   // ~4 GB written per variant
      const double bytes = nval*sizeof(float_sw4);
      for( int v=0 ; v < nvar ; v++ )
      {
         // check: garbage in, only zeros out
#pragma omp parallel for
         for( size_t i=0 ; i < nval ; i++ )
            a.data[i] = 1.0f;
         fns[v](a);
         size_t bad = 0;
#pragma omp parallel for reduction(+:bad)
         for( size_t i=0 ; i < nval ; i++ )
            bad += a.data[i] != 0;

         double tmin = 1e30, tsum = 0;
         for( int r=0 ; r < reps ; r++ )
         {
            double t0 = omp_get_wtime();
            fns[v](a);
            double t = omp_get_wtime() - t0;
            tmin = std::min(tmin, t);
            tsum += t;
         }
         printf("%6d %10.1f %-10s %10.3f %10.3f %9.2f %s\n", n, bytes/1e6, names[v],
                1e3*tmin, 1e3*tsum/reps, bytes/tmin/1e9, bad ? "NONZERO" : "ok");
      }
      fflush(stdout);
      delete[] a.data;
   }
   return 0;
}
