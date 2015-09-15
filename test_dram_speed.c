#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include "nt_memcpy.h"

int main(int argc, char *argv[])
{

  int nthread=1;
  if (argc>1)
    nthread=atoi(argv[1]);
  omp_set_num_threads(nthread);
#pragma omp parallel
#pragma omp master
  printf("have %d threads.\n",omp_get_num_threads());


  double tot_bw=0;

#pragma omp parallel
  {
  printf("char size is %d\n",sizeof(char));

  long nelem=1024*1024*512;
  char *a=(char *)malloc(nelem*sizeof(char));
  char *b=(char *)malloc(nelem*sizeof(char));

  memset(a,0,nelem*sizeof(char));


  double t_start=omp_get_wtime();
  int niter=50;
  double start1;
  for (int i=0;i<niter;i++) {
    if (i==1)
      start1=omp_get_wtime();
    double t0=omp_get_wtime();
#if 1
    //memcpy(b,a,sizeof(char)*nelem);
    nt_memcpy(b,a,sizeof(char)*nelem);

#else
    for (int j=0;j<nelem;j++)
      b[j]=a[j];
#endif

    double dt=omp_get_wtime()-t0;
    //printf("elapsed time %3d is %12.4f\n",i,omp_get_wtime()-t0);
    printf("Bandwith on transfer %2d is %12.4f MB/s %d\n",i,nelem/(dt*1024*1024),b[0]);
  }
  double t_stop=omp_get_wtime();
  //skip the first iteration since that often is quite slow
  double mybw=((niter-1)*nelem)/(t_stop-start1)/1024/1024;
  printf("Thread %d averaged %12.4f MB/s\n",omp_get_thread_num(),mybw);
  #pragma omp critical
  tot_bw+=mybw;
  }
  printf("total aggregate bandwidth was %12.4f MB/s\n",tot_bw);


}
