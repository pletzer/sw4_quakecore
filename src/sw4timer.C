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
const char* const mpi_fn_names[sw4timer::MPIFN_COUNT] = {
   "MPI_Sendrecv", "MPI_Send", "MPI_Recv", "MPI_Isend", "MPI_Irecv",
   "MPI_Wait", "MPI_Waitall", "MPI_Barrier", "MPI_Bcast", "MPI_Reduce",
   "MPI_Allreduce", "MPI_Gather", "MPI_Gatherv", "MPI_Allgather",
   "MPI_Allgatherv" };

struct MpiStat
{
   double t = 0, bytes = 0, msgs = 0, calls = 0;
   void add( double s, double b, int m )
   {
      t += s;
      bytes += b;
      msgs += m;
      calls += 1;
   }
};

struct Node
{
   const char* name;
   int parent;
   std::vector<int> children;
   double tstart;
   double total;
   long calls;
   MpiStat mpi; // exclusive: MPI calls made while this region was innermost
};

// Node 0 is the root; it is never timed, but collects the MPI calls made
// outside any region. stack holds the running regions.
struct Registry
{
   std::vector<Node> nodes;
   std::vector<int> stack;
   MpiStat fn[sw4timer::MPIFN_COUNT];
   double t0;
   bool started;
   bool warned;
   bool in_report;
   Registry() { clear(); }
   void clear()
   {
      nodes.assign( 1, Node{ "", -1, {}, 0.0, 0.0, 0, MpiStat() } );
      stack.assign( 1, 0 );
      for( auto& f : fn )
         f = MpiStat();
      t0 = 0;
      started = false;
      warned = false;
      in_report = false;
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

// MPI totals of a region including all of its sub-regions.
MpiStat inclusive_mpi( const Registry& r, int n, std::vector<MpiStat>& inc )
{
   MpiStat s = r.nodes[n].mpi;
   for( int c : r.nodes[n].children )
   {
      MpiStat cs = inclusive_mpi( r, c, inc );
      s.t += cs.t;
      s.bytes += cs.bytes;
      s.msgs += cs.msgs;
      s.calls += cs.calls;
   }
   inc[n] = s;
   return s;
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

// Per-region fields reduced over ranks with MIN, MAX and SUM.
enum Field { F_TIME, F_CALLS, F_PRESENT, F_MPI_T, F_MPI_BYTES, F_MPI_MSGS, F_MPI_CALLS, NF };

void reduce3( std::vector<double>& mn, std::vector<double>& mx, std::vector<double>& sm,
              int rank, MPI_Comm comm )
{
   const int n = static_cast<int>( mn.size() );
   if( n == 0 )
      return;
   MPI_Reduce( rank == 0 ? MPI_IN_PLACE : mn.data(), mn.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm );
   MPI_Reduce( rank == 0 ? MPI_IN_PLACE : mx.data(), mx.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm );
   MPI_Reduce( rank == 0 ? MPI_IN_PLACE : sm.data(), sm.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm );
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
      r.nodes.push_back( Node{ name, cur, {}, 0.0, 0.0, 0, MpiStat() } );
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

void add_mpi( int fn, double seconds, double bytes, int messages )
{
   Registry& r = reg();
   if( r.in_report || in_parallel() || fn < 0 || fn >= MPIFN_COUNT )
      return;
   r.nodes[r.stack.back()].mpi.add( seconds, bytes, messages );
   r.fn[fn].add( seconds, bytes, messages );
}

void reset()
{
   reg().clear();
}

void report( MPI_Comm comm, std::ostream& os )
{
   Registry& r = reg();
   r.in_report = true; // the reductions below are not part of the run
   int rank, nprocs;
   MPI_Comm_rank( comm, &rank );
   MPI_Comm_size( comm, &nprocs );

   const double now = MPI_Wtime();
   const double wall = r.started ? now - r.t0 : 0.0;

// Regions still running are reported up to now, without counting the call.
   std::vector<double> running( r.nodes.size(), 0.0 );
   for( size_t s = 1; s < r.stack.size(); s++ )
      running[r.stack[s]] = now - r.nodes[r.stack[s]].tstart;

   std::vector<MpiStat> incmpi( r.nodes.size() );
   inclusive_mpi( r, 0, incmpi );

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

// Row np holds the MPI calls made outside any region (root node), rows
// np+1 .. np+MPIFN_COUNT the per-function totals.
   const int nrows = np + 1 + MPIFN_COUNT;
   const double huge = std::numeric_limits<double>::max();
   std::vector<double> vmin( nrows * NF ), vmax( nrows * NF ), vsum( nrows * NF );
   auto put = []( std::vector<double>& v, int row, double t, double c, double pr, const MpiStat& m )
   {
      double* f = &v[row * NF];
      f[F_TIME] = t;
      f[F_CALLS] = c;
      f[F_PRESENT] = pr;
      f[F_MPI_T] = m.t;
      f[F_MPI_BYTES] = m.bytes;
      f[F_MPI_MSGS] = m.msgs;
      f[F_MPI_CALLS] = m.calls;
   };
   for( int i = 0; i < np; i++ )
   {
      auto it = local.find( paths[i] );
      if( it != local.end() )
      {
         const int id = it->second;
         const double t = r.nodes[id].total + running[id];
         const double c = static_cast<double>( r.nodes[id].calls );
         put( vmin, i, t, c, 1, incmpi[id] );
         put( vmax, i, t, c, 1, incmpi[id] );
         put( vsum, i, t, c, 1, incmpi[id] );
      }
      else
      {
         MpiStat z;
         put( vmax, i, 0, 0, 0, z );
         put( vsum, i, 0, 0, 0, z );
         z.t = z.bytes = z.msgs = z.calls = huge;
         put( vmin, i, huge, huge, 0, z );
      }
   }
   for( auto* v : { &vmin, &vmax, &vsum } )
   {
      put( *v, np, 0, 0, 1, r.nodes[0].mpi );
      for( int f = 0; f < MPIFN_COUNT; f++ )
         put( *v, np + 1 + f, 0, 0, 1, r.fn[f] );
   }
   reduce3( vmin, vmax, vsum, rank, comm );

   double wmax = 0, wsum = 0;
   MPI_Reduce( &wall, &wmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
   MPI_Reduce( &wall, &wsum, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
   r.in_report = false;

   if( rank != 0 )
      return;

   const double wavg = wsum / nprocs;
   auto ranks_of = [&]( int row ) { return static_cast<int>( vsum[row * NF + F_PRESENT] + 0.5 ); };
   auto avg_of = [&]( int row, int f ) {
      const int k = ranks_of( row );
      return k > 0 ? vsum[row * NF + f] / k : 0.0;
   };
   auto min_of = [&]( int row, int f ) { return ranks_of( row ) > 0 ? vmin[row * NF + f] : 0.0; };
   auto max_of = [&]( int row, int f ) { return vmax[row * NF + f]; };

// Indented leaf names; the full path goes to the CSV.
   std::vector<std::string> labels( np );
   std::vector<int> depth( np );
   size_t width = 22;
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
      const double avg = avg_of( i, F_TIME );
      o << std::left << std::setw( width ) << labels[i] << std::right
        << std::setw( 10 ) << static_cast<long>( max_of( i, F_CALLS ) + 0.5 )
        << std::setw( 7 ) << ranks_of( i )
        << std::fixed << std::setprecision( 4 )
        << std::setw( 12 ) << min_of( i, F_TIME ) << std::setw( 12 ) << avg
        << std::setw( 12 ) << max_of( i, F_TIME )
        << std::setprecision( 2 ) << std::setw( 8 ) << ( avg > 0 ? max_of( i, F_TIME ) / avg : 1.0 )
        << std::setprecision( 1 ) << std::setw( 9 ) << ( wavg > 0 ? 100.0 * avg / wavg : 0.0 )
        << "\n";
   }
   o << std::string( line, '-' ) << "\n";

// MPI tables, only if the wrappers saw anything.
   double mpi_calls_all = 0;
   for( int f = 0; f < MPIFN_COUNT; f++ )
      mpi_calls_all += vsum[( np + 1 + f ) * NF + F_MPI_CALLS];
   if( mpi_calls_all > 0 )
   {
      const int mline = static_cast<int>( width ) + 87;
      o << "\n" << std::string( mline, '-' ) << "\n"
        << "  MPI communication per region, including sub-regions; time in seconds, min/avg/max over ranks\n"
        << "  msgs and MB = sent per rank (avg), KB/msg = avg payload per message,\n"
        << "  %reg = MPI share of the region's time; MPI time includes waiting for slower ranks\n"
        << std::string( mline, '-' ) << "\n";
      o << std::left << std::setw( width ) << "Region" << std::right
        << std::setw( 11 ) << "msgs" << std::setw( 11 ) << "MB"
        << std::setw( 9 ) << "KB/msg" << std::setw( 12 ) << "min"
        << std::setw( 12 ) << "avg" << std::setw( 12 ) << "max"
        << std::setw( 8 ) << "imb" << std::setw( 8 ) << "%reg"
        << std::setw( 9 ) << "%wall" << "\n";
      auto mpi_row = [&]( const std::string& label, int row, double region_t )
      {
         const double mavg = avg_of( row, F_MPI_T );
         const double msgs = avg_of( row, F_MPI_MSGS );
         const double bytes = avg_of( row, F_MPI_BYTES );
         o << std::left << std::setw( width ) << label << std::right << std::fixed
           << std::setprecision( 0 ) << std::setw( 11 ) << msgs
           << std::setprecision( 2 ) << std::setw( 11 ) << bytes / 1048576.0
           << std::setprecision( 2 ) << std::setw( 9 ) << ( msgs > 0 ? bytes / msgs / 1024.0 : 0.0 )
           << std::setprecision( 4 ) << std::setw( 12 ) << min_of( row, F_MPI_T )
           << std::setw( 12 ) << mavg << std::setw( 12 ) << max_of( row, F_MPI_T )
           << std::setprecision( 2 ) << std::setw( 8 ) << ( mavg > 0 ? max_of( row, F_MPI_T ) / mavg : 1.0 )
           << std::setprecision( 1 ) << std::setw( 8 ) << ( region_t > 0 ? 100.0 * mavg / region_t : 0.0 )
           << std::setw( 9 ) << ( wavg > 0 ? 100.0 * mavg / wavg : 0.0 ) << "\n";
      };
      for( int i = 0; i < np; i++ )
         if( max_of( i, F_MPI_CALLS ) > 0 )
            mpi_row( labels[i], i, avg_of( i, F_TIME ) );
      if( max_of( np, F_MPI_CALLS ) > 0 )
         mpi_row( "(outside any region)", np, 0.0 );
      o << std::string( mline, '-' ) << "\n";

      o << "\n" << std::left << std::setw( width ) << "MPI function" << std::right
        << std::setw( 11 ) << "calls" << std::setw( 11 ) << "MB"
        << std::setw( 9 ) << "KB/msg" << std::setw( 12 ) << "min"
        << std::setw( 12 ) << "avg" << std::setw( 12 ) << "max"
        << std::setw( 8 ) << "imb" << std::setw( 8 ) << "" << std::setw( 9 ) << "%wall" << "\n";
      double tot_t = 0, tot_b = 0, tot_m = 0, tot_c = 0;
      for( int f = 0; f < MPIFN_COUNT; f++ )
      {
         const int row = np + 1 + f;
         if( max_of( row, F_MPI_CALLS ) == 0 )
            continue;
         const double mavg = avg_of( row, F_MPI_T );
         const double msgs = avg_of( row, F_MPI_MSGS );
         const double bytes = avg_of( row, F_MPI_BYTES );
         tot_t += mavg;
         tot_b += bytes;
         tot_m += msgs;
         tot_c += avg_of( row, F_MPI_CALLS );
         o << std::left << std::setw( width ) << mpi_fn_names[f] << std::right << std::fixed
           << std::setprecision( 0 ) << std::setw( 11 ) << avg_of( row, F_MPI_CALLS )
           << std::setprecision( 2 ) << std::setw( 11 ) << bytes / 1048576.0
           << std::setw( 9 ) << ( msgs > 0 ? bytes / msgs / 1024.0 : 0.0 )
           << std::setprecision( 4 ) << std::setw( 12 ) << min_of( row, F_MPI_T )
           << std::setw( 12 ) << mavg << std::setw( 12 ) << max_of( row, F_MPI_T )
           << std::setprecision( 2 ) << std::setw( 8 ) << ( mavg > 0 ? max_of( row, F_MPI_T ) / mavg : 1.0 )
           << std::setw( 8 ) << ""
           << std::setprecision( 1 ) << std::setw( 9 ) << ( wavg > 0 ? 100.0 * mavg / wavg : 0.0 ) << "\n";
      }
      o << std::left << std::setw( width ) << "total (avg per rank)" << std::right << std::fixed
        << std::setprecision( 0 ) << std::setw( 11 ) << tot_c
        << std::setprecision( 2 ) << std::setw( 11 ) << tot_b / 1048576.0
        << std::setw( 9 ) << ( tot_m > 0 ? tot_b / tot_m / 1024.0 : 0.0 )
        << std::setw( 12 ) << "" << std::setprecision( 4 ) << std::setw( 12 ) << tot_t
        << std::setw( 12 ) << "" << std::setw( 8 ) << "" << std::setw( 8 ) << ""
        << std::setprecision( 1 ) << std::setw( 9 ) << ( wavg > 0 ? 100.0 * tot_t / wavg : 0.0 ) << "\n";
      o << std::string( mline, '-' ) << "\n";
   }
   os << o.str() << std::flush;

   if( const char* csv = std::getenv( "SW4_TIMERS_CSV" ) )
   {
      std::ofstream f( csv, std::ofstream::trunc );
      if( f )
      {
         f << "region,depth,calls_max,calls_total,ranks,min_s,avg_s,max_s,wall_avg_s,"
              "mpi_min_s,mpi_avg_s,mpi_max_s,mpi_bytes_avg,mpi_msgs_avg,mpi_calls_avg\n";
         f << std::setprecision( 9 );
         auto csv_row = [&]( const std::string& name, int d, int row )
         {
            f << name << "," << d << "," << static_cast<long>( max_of( row, F_CALLS ) + 0.5 ) << ","
              << static_cast<long>( vsum[row * NF + F_CALLS] + 0.5 ) << "," << ranks_of( row ) << ","
              << min_of( row, F_TIME ) << "," << avg_of( row, F_TIME ) << "," << max_of( row, F_TIME ) << ","
              << wavg << "," << min_of( row, F_MPI_T ) << "," << avg_of( row, F_MPI_T ) << ","
              << max_of( row, F_MPI_T ) << "," << avg_of( row, F_MPI_BYTES ) << ","
              << avg_of( row, F_MPI_MSGS ) << "," << avg_of( row, F_MPI_CALLS ) << "\n";
         };
         for( int i = 0; i < np; i++ )
            csv_row( paths[i], depth[i], i );
         csv_row( "(outside any region)", -1, np );
         for( int fn = 0; fn < MPIFN_COUNT; fn++ )
            if( max_of( np + 1 + fn, F_MPI_CALLS ) > 0 )
               csv_row( mpi_fn_names[fn], -1, np + 1 + fn );
      }
      else
         std::cerr << "sw4timer: cannot open SW4_TIMERS_CSV=" << csv << std::endl;
   }
}
} // namespace sw4timer
