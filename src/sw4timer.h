// Lightweight hierarchical wall-clock region timers.
//
// Usage:
//    { SW4_TIME("evalRHS"); ... }                 // RAII, stops at end of scope
//    SW4_TIMER_START("comm"); ... SW4_TIMER_STOP("comm");  // explicit pair
//    sw4timer::report( MPI_COMM_WORLD, std::cout );       // collective, at the end
//
// Regions nest: a region started while another is running is recorded as its
// child, so the same name under different parents ("predictor/evalRHS" and
// "corrector/evalRHS") is accumulated separately. report() reduces over all
// ranks of the communicator and prints, per region, the number of calls and
// the min/avg/max time over ranks, the load imbalance max/avg and the share of
// the run's wall time. Ranks need not execute the same regions.
//
// Region names must be string literals (or otherwise outlive the run): only
// the pointer is stored. Calls made inside an OpenMP parallel region are
// ignored -- time the parallel region from the outside.
//
// Cost is one hash-free child lookup and one MPI_Wtime() per start/stop, so put
// regions around kernels and phases, never inside grid-point loops.
// Configure with -DSW4_TIMERS=OFF (CMake) or define SW4_NO_TIMERS to compile
// every SW4_TIME* macro to nothing.
//
// Set SW4_TIMERS_CSV=<file> in the environment to also write the table as CSV.
#ifndef SW4_TIMER_H
#define SW4_TIMER_H

#include <mpi.h>
#include <iosfwd>

namespace sw4timer
{
   void start( const char* name );
   void stop( const char* name );
   void report( MPI_Comm comm, std::ostream& os );
   void reset();

   class Scope
   {
   public:
      explicit Scope( const char* name ) : m_name(name) { start(name); }
      ~Scope() { stop(m_name); }
      Scope( const Scope& ) = delete;
      Scope& operator=( const Scope& ) = delete;
   private:
      const char* m_name;
   };
}

#define SW4_TIMER_CAT2(a,b) a##b
#define SW4_TIMER_CAT(a,b) SW4_TIMER_CAT2(a,b)

#ifndef SW4_NO_TIMERS
#define SW4_TIME(name) sw4timer::Scope SW4_TIMER_CAT(sw4_timer_scope_,__LINE__)(name)
#define SW4_TIMER_START(name) sw4timer::start(name)
#define SW4_TIMER_STOP(name) sw4timer::stop(name)
#define SW4_TIMER_REPORT(comm,os) sw4timer::report(comm,os)
#else
#define SW4_TIME(name) do {} while(0)
#define SW4_TIMER_START(name) do {} while(0)
#define SW4_TIMER_STOP(name) do {} while(0)
#define SW4_TIMER_REPORT(comm,os) do {} while(0)
#endif

#endif
