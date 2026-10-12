#include "sw4timer.h"

#include <mpi.h>
#ifndef SW4_NOOMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
struct Node
{
   const char* name;
   int parent;
   std::vector<int> children;
   double tstart;
   double total;
   long calls;
};

// Node 0 is the root; it is never timed. m_stack holds the running regions.
struct Registry
{
   std::vector<Node> nodes;
   std::vector<int> stack;
   double t0;
   bool started;
   bool warned;
   Registry() { clear(); }
   void clear()
   {
      nodes.assign( 1, Node{ "", -1, {}, 0.0, 0.0, 0 } );
      stack.assign( 1, 0 );
      t0 = 0;
      started = false;
      warned = false;
   }
};

Registry& reg()
{
   static Registry r;
   return r;
}

bool in_parallel()
{
#ifndef SW4_NOOMP
   return omp_in_parallel() != 0;
#else
   return false;
#endif
}

bool same_name( const char* a, const char* b )
{
   return a == b || std::strcmp( a, b ) == 0;
}

// Path of every node, depth first in first-call order, "a/b/c".
void collect_paths( const Registry& r, int n, const std::string& prefix,
                    std::vector<std::string>& paths, std::vector<int>& ids )
{
   for( int c : r.nodes[n].children )
   {
      std::string p = prefix.empty() ? r.nodes[c].name : prefix + "/" + r.nodes[c].name;
      paths.push_back( p );
      ids.push_back( c );
      collect_paths( r, c, p, paths, ids );
   }
}

std::string join( const std::vector<std::string>& v )
{
   std::string s;
   for( const auto& x : v )
   {
      s += x;
      s += '\n';
   }
   return s;
}

std::vector<std::string> split( const std::string& s, char sep )
{
   std::vector<std::string> v;
   std::string cur;
   for( char ch : s )
   {
      if( ch == sep )
      {
         v.push_back( cur );
         cur.clear();
      }
      else
         cur += ch;
   }
   if( !cur.empty() )
      v.push_back( cur );
   return v;
}

// Merge the region paths of all ranks into one tree and return it depth first,
// so a region that only some ranks execute still lands under its parent.
struct UNode
{
   std::string name;
   std::vector<int> kids;
};

int child_of( std::vector<UNode>& t, int n, const std::string& name )
{
   for( int k : t[n].kids )
      if( t[k].name == name )
         return k;
   t.push_back( UNode{ name, {} } );
   int k = static_cast<int>( t.size() ) - 1;
   t[n].kids.push_back( k );
   return k;
}

void flatten( const std::vector<UNode>& t, int n, const std::string& prefix,
              std::vector<std::string>& out )
{
   for( int k : t[n].kids )
   {
      std::string p = prefix.empty() ? t[k].name : prefix + "/" + t[k].name;
      out.push_back( p );
      flatten( t, k, p, out );
   }
}
} // namespace

namespace sw4timer
{
void start( const char* name )
{
   if( in_parallel() )
      return;
   Registry& r = reg();
   double now = MPI_Wtime();
   if( !r.started )
   {
      r.t0 = now;
      r.started = true;
   }
   int cur = r.stack.back();
   int found = -1;
   for( int c : r.nodes[cur].children )
      if( same_name( r.nodes[c].name, name ) )
      {
         found = c;
         break;
      }
   if( found < 0 )
   {
      r.nodes.push_back( Node{ name, cur, {}, 0.0, 0.0, 0 } );
      found = static_cast<int>( r.nodes.size() ) - 1;
      r.nodes[cur].children.push_back( found );
   }
   r.nodes[found].tstart = now;
   r.stack.push_back( found );
}

void stop( const char* name )
{
   if( in_parallel() )
      return;
   double now = MPI_Wtime();
   Registry& r = reg();
   if( r.stack.size() <= 1 )
   {
      if( !r.warned )
         std::cerr << "sw4timer: stop(\"" << name << "\") without a running region" << std::endl;
      r.warned = true;
      return;
   }
   int n = r.stack.back();
   if( !same_name( r.nodes[n].name, name ) && !r.warned )
   {
      std::cerr << "sw4timer: stop(\"" << name << "\") while \"" << r.nodes[n].name
                << "\" is running; regions are not properly nested" << std::endl;
      r.warned = true;
   }
   r.nodes[n].total += now - r.nodes[n].tstart;
   r.nodes[n].calls++;
   r.stack.pop_back();
}

void reset()
{
   reg().clear();
}

void report( MPI_Comm comm, std::ostream& os )
{
   Registry& r = reg();
   int rank, nprocs;
   MPI_Comm_rank( comm, &rank );
   MPI_Comm_size( comm, &nprocs );

   const double now = MPI_Wtime();
   const double wall = r.started ? now - r.t0 : 0.0;

// Regions still running are reported up to now, without counting the call.
   std::vector<double> running( r.nodes.size(), 0.0 );
   for( size_t s = 1; s < r.stack.size(); s++ )
      running[r.stack[s]] = now - r.nodes[r.stack[s]].tstart;

   std::vector<std::string> mypaths;
   std::vector<int> myids;
   collect_paths( r, 0, "", mypaths, myids );

// Gather everybody's region list on rank 0 and merge.
   std::string mine = join( mypaths );
   int mylen = static_cast<int>( mine.size() );
   std::vector<int> lens( nprocs ), displs( nprocs, 0 );
   MPI_Gather( &mylen, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, comm );
   std::string all;
   if( rank == 0 )
   {
      for( int p = 1; p < nprocs; p++ )
         displs[p] = displs[p - 1] + lens[p - 1];
      all.resize( displs[nprocs - 1] + lens[nprocs - 1] );
   }
   MPI_Gatherv( mine.data(), mylen, MPI_CHAR, rank == 0 ? &all[0] : nullptr,
                lens.data(), displs.data(), MPI_CHAR, 0, comm );

   std::string merged;
   if( rank == 0 )
   {
      std::vector<UNode> tree( 1 );
      for( const auto& path : split( all, '\n' ) )
      {
         int n = 0;
         for( const auto& part : split( path, '/' ) )
            n = child_of( tree, n, part );
      }
      std::vector<std::string> flat;
      flatten( tree, 0, "", flat );
      merged = join( flat );
   }
   int mlen = static_cast<int>( merged.size() );
   MPI_Bcast( &mlen, 1, MPI_INT, 0, comm );
   merged.resize( mlen );
   MPI_Bcast( &merged[0], mlen, MPI_CHAR, 0, comm );
   const std::vector<std::string> paths = split( merged, '\n' );
   const int np = static_cast<int>( paths.size() );

   std::map<std::string, int> local;
   for( size_t i = 0; i < mypaths.size(); i++ )
      local[mypaths[i]] = myids[i];

   const double huge = std::numeric_limits<double>::max();
   std::vector<double> tmin( np ), tmax( np ), tsum( np ), cmax( np ), csum( np ), nr( np );
   for( int i = 0; i < np; i++ )
   {
      auto it = local.find( paths[i] );
      if( it != local.end() )
      {
         const Node& n = r.nodes[it->second];
         double t = n.total + running[it->second];
         tmin[i] = tmax[i] = tsum[i] = t;
         cmax[i] = csum[i] = static_cast<double>( n.calls );
         nr[i] = 1;
      }
      else
      {
         tmin[i] = huge;
         tmax[i] = tsum[i] = cmax[i] = csum[i] = nr[i] = 0;
      }
   }
   double wmax = 0, wsum = 0;
   if( np > 0 )
   {
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : tmin.data(), tmin.data(), np, MPI_DOUBLE, MPI_MIN, 0, comm );
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : tmax.data(), tmax.data(), np, MPI_DOUBLE, MPI_MAX, 0, comm );
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : tsum.data(), tsum.data(), np, MPI_DOUBLE, MPI_SUM, 0, comm );
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : cmax.data(), cmax.data(), np, MPI_DOUBLE, MPI_MAX, 0, comm );
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : csum.data(), csum.data(), np, MPI_DOUBLE, MPI_SUM, 0, comm );
      MPI_Reduce( rank == 0 ? MPI_IN_PLACE : nr.data(), nr.data(), np, MPI_DOUBLE, MPI_SUM, 0, comm );
   }
   MPI_Reduce( &wall, &wmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
   MPI_Reduce( &wall, &wsum, 1, MPI_DOUBLE, MPI_SUM, 0, comm );

   if( rank != 0 )
      return;

   const double wavg = wsum / nprocs;

// Indented leaf names; the full path goes to the CSV.
   std::vector<std::string> labels( np );
   std::vector<int> depth( np );
   size_t width = 6;
   for( int i = 0; i < np; i++ )
   {
      std::vector<std::string> parts = split( paths[i], '/' );
      depth[i] = static_cast<int>( parts.size() ) - 1;
      labels[i] = std::string( 2 * depth[i], ' ' ) + parts.back();
      width = std::max( width, labels[i].size() );
   }

   std::ostringstream o;
   const int line = static_cast<int>( width ) + 74;
   o << "\n" << std::string( line, '-' ) << "\n"
     << "  SW4 region timers: wall time in seconds over " << nprocs << " rank"
     << ( nprocs > 1 ? "s" : "" ) << ", min/avg/max over ranks\n"
     << "  calls = per rank (max), imb = max/avg, %wall = avg / avg wall time ("
     << std::fixed << std::setprecision( 3 ) << wavg << " s)\n"
     << std::string( line, '-' ) << "\n";
   o << std::left << std::setw( width ) << "Region" << std::right
     << std::setw( 10 ) << "calls" << std::setw( 7 ) << "ranks"
     << std::setw( 12 ) << "min" << std::setw( 12 ) << "avg"
     << std::setw( 12 ) << "max" << std::setw( 8 ) << "imb"
     << std::setw( 9 ) << "%wall" << "\n";
   for( int i = 0; i < np; i++ )
   {
      const int ranks = static_cast<int>( nr[i] + 0.5 );
      const double avg = ranks > 0 ? tsum[i] / ranks : 0.0;
      const double mn = ranks > 0 ? tmin[i] : 0.0;
      o << std::left << std::setw( width ) << labels[i] << std::right
        << std::setw( 10 ) << static_cast<long>( cmax[i] + 0.5 )
        << std::setw( 7 ) << ranks
        << std::fixed << std::setprecision( 4 )
        << std::setw( 12 ) << mn << std::setw( 12 ) << avg << std::setw( 12 ) << tmax[i]
        << std::setprecision( 2 ) << std::setw( 8 ) << ( avg > 0 ? tmax[i] / avg : 1.0 )
        << std::setprecision( 1 ) << std::setw( 9 ) << ( wavg > 0 ? 100.0 * avg / wavg : 0.0 )
        << "\n";
   }
   o << std::string( line, '-' ) << "\n";
   os << o.str() << std::flush;

   if( const char* csv = std::getenv( "SW4_TIMERS_CSV" ) )
   {
      std::ofstream f( csv, std::ofstream::trunc );
      if( f )
      {
         f << "region,depth,calls_max,calls_total,ranks,min_s,avg_s,max_s,wall_avg_s\n";
         f << std::setprecision( 9 );
         for( int i = 0; i < np; i++ )
         {
            const int ranks = static_cast<int>( nr[i] + 0.5 );
            f << paths[i] << "," << depth[i] << "," << static_cast<long>( cmax[i] + 0.5 ) << ","
              << static_cast<long>( csum[i] + 0.5 ) << "," << ranks << ","
              << ( ranks > 0 ? tmin[i] : 0.0 ) << "," << ( ranks > 0 ? tsum[i] / ranks : 0.0 ) << ","
              << tmax[i] << "," << wavg << "\n";
         }
      }
      else
         std::cerr << "sw4timer: cannot open SW4_TIMERS_CSV=" << csv << std::endl;
   }
}
} // namespace sw4timer
