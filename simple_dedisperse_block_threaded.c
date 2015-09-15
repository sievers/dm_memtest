//    Copyright Jonathan Sievers, 2015.  All rights reserved.  This code may only be used with permission of the owner.                                                           


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <emmintrin.h>

float **matrix(int n, int m)
{
  float *vec=(float *)malloc(sizeof(float)*n*m);
  float **mat=(float **)malloc(sizeof(float *)*n);
  memset(vec,0,n*m*sizeof(float));
  for (int i=0;i<n;i++)
    mat[i]=vec+i*m;
  return mat;
}

/*--------------------------------------------------------------------------------*/

void dedisperse_kernel(float **in, float **out, int n, int m)
{

  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<m;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=0;i<m-jj-1;i++) 
      out[jj+npair][i]=in[2*jj][i+jj]+in[2*jj+1][i+jj+1];
    
  }
}
/*--------------------------------------------------------------------------------*/

void dedisperse_kernel_v2(float **in, float **out, int n, int m)
{

  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<jj;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=jj;i<m-1;i++) {
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
      out[jj+npair][i-jj]=in[2*jj][i]+in[2*jj+1][i+1];
    }
    
  }
}
/*--------------------------------------------------------------------------------*/

int get_npass(int n)
{
  int nn=0;
  while (n>1) {
    nn++;
    n/=2;
  }
  return nn;
}
/*--------------------------------------------------------------------------------*/
void dedisperse(float **inin, float **outout, int nchan,int ndat)
{
  //return;
  int npass=get_npass(nchan);
  //printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  for (int i=0;i<npass;i++) {    
    //#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      dedisperse_kernel(in+j,out+j,bs,ndat);
    }
    bs/=2;
    float **tmp=in;
    in=out;
    out=tmp;
  }
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}
/*--------------------------------------------------------------------------------*/
void dedisperse_2pass(float **dat, float **dat2, int nchan, int ndat)
{
  dedisperse_kernel(dat,dat2,nchan,ndat);
  dedisperse_kernel(dat2,dat,nchan/2,ndat);
  dedisperse_kernel(dat2+nchan/2,dat+nchan/2,nchan/2,ndat);
}

/*--------------------------------------------------------------------------------*/

void write_mat(float **dat, int n, int m, char *fname)
{
  FILE *outfile=fopen(fname,"w");
  fwrite(&n,sizeof(int),1,outfile);
  fwrite(&m,sizeof(int),1,outfile);
  fwrite(dat[0],sizeof(float),n*m,outfile);
  fclose(outfile);
}

/*--------------------------------------------------------------------------------*/
void find_peak(float **dat, int nchan, int ndat, int *best_chan, int *best_dat)
{
  float max=0;
  int ichan=0;
  int idat=0;
  for (int i=0;i<nchan;i++) 
    for (int j=0;j<ndat;j++) {
      if (dat[i][j]>max) {
	max=dat[i][j];
	ichan=i;
	idat=j;
      }
    }
  float slope=(float)ichan/(float)nchan;
  float flux=dat[ichan][idat]+dat[ichan][idat-1]+dat[ichan][idat+1];
  printf("I found a peak at slope %8.4f at time %d with average flux %8.4f\n",slope,idat,flux/(float)nchan);

  if (best_chan)
    *best_chan=ichan;
  if (best_dat)
    *best_dat=idat;

}
/*--------------------------------------------------------------------------------*/
void dedisperse_blocked(float **dat, float **dat2, int nchan, int ndat)
{
  int nchan1=64;
  int npass1=get_npass(nchan1);
  int npass=get_npass(nchan);
  int npass2=npass-npass1;
  int nchan2=nchan/nchan1;

  int nblock=nchan/nchan1;
  int nblock2=nchan/nchan2;

  for (int i=0;i<nblock;i++) 
    dedisperse(dat+i*nchan1,dat2+i*nchan1,nchan1,ndat);
  
  
  for (int i=0;i<nblock;i++) 
    for (int j=0;j<nchan1;j++)
      memcpy(dat2[j*nblock+i],dat[i*nchan1+j]+i*j,ndat-i*j);

  for (int i=0;i<nblock2;i++)
    dedisperse(dat2+i*nchan2,dat+i*nchan2,nchan2,ndat);
  
  
}
/*--------------------------------------------------------------------------------*/
void dedisperse_blocked_cached(float **dat, float **dat2, int nchan, int ndat)
{
  //int nchan1=128;
  //int chunk_size=768;

  int nchan1=64;
  int chunk_size=1536;
  int nchunk=ndat/chunk_size;
  int npass1=get_npass(nchan1);
  int npass=get_npass(nchan);
  int npass2=npass-npass1;
  int nchan2=nchan/nchan1;

  int nblock=nchan/nchan1;
  int nblock2=nchan/nchan2;


  
#pragma omp parallel 
  {
    float **tmp1=matrix(nchan1,chunk_size+nchan1);
    float **tmp2=matrix(nchan1,chunk_size+nchan1); 
    
#pragma omp for collapse(2) schedule(dynamic,2)
    for (int i=0;i<nblock;i++) {      
      //printf("i is %d\n",i);
      for (int j=0;j<nchunk;j++) {
	int istart=j*chunk_size;
	int istop=(j+1)*chunk_size+nchan1;
	if (istop>ndat) {
	  istop=ndat;
	  for (int k=0;k<nchan1;k++)
	    memset(tmp1[k]+chunk_size,0,sizeof(float)*nchan1);
	}
	for (int k=0;k<nchan1;k++)
	  memcpy(tmp1[k],&(dat[i*nchan1+k][istart]),(istop-istart)*sizeof(float));
	
#ifndef NO_DEDISPERSE	
	dedisperse(tmp1,tmp2,nchan1,chunk_size+nchan1);
#endif
	
	for (int k=0;k<nchan1;k++)
	  memcpy(&(dat2[i*nchan1+k][istart]),tmp1[k],chunk_size*sizeof(float));
      }
    }
#if 1
    free(tmp1[0]);   
    free(tmp1);
    free(tmp2[0]);
    free(tmp2);
#endif
  }
  
  

  float **dat_shift=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nblock;i++)
    for (int j=0;j<nchan1;j++)
      dat_shift[j*nblock+i]=dat2[i*nchan1+j]+i*j;  


  //recalculate block sizes to keep amount in cache about the same
  int nelem=nchan1*chunk_size;
  chunk_size=nelem/nchan2;
  nchunk=ndat/chunk_size;

#pragma omp parallel 
  {
    float **tmp1=matrix(nchan2,chunk_size+nchan2);
    float **tmp2=matrix(nchan2,chunk_size+nchan2); 

#pragma omp for  collapse(2) schedule(dynamic,4)
    for (int i=0;i<nblock2;i++) {      
      //printf("i is now %d\n",i);
      for (int j=0;j<nchunk;j++) {
	int istart=j*chunk_size;
	int istop=(j+1)*chunk_size+nchan2;
	if (istop>ndat) {
	  istop=ndat;
	  for (int k=0;k<nchan2;k++)
	    memset(tmp1[k]+chunk_size,0,sizeof(float)*nchan2);
	}
	for (int k=0;k<nchan2;k++) {
	  memcpy(tmp1[k],dat_shift[i*nchan2+k]+istart,(istop-istart)*sizeof(float));
	}
#ifndef NO_DEDISPERSE	
	dedisperse(tmp1,tmp2,nchan2,chunk_size+nchan2);
#endif

	for (int k=0;k<nchan2;k++)
	  memcpy(dat[i*nchan2+k]+istart,tmp2[k],chunk_size*sizeof(float));
      }
    }
    free(tmp1[0]);
    free(tmp1);
    free(tmp2[0]);
    free(tmp2);
    
  }
  //printf("Finished dedispersion.\n");
}

/*================================================================================*/

int main(int argc, char *argv[])
{
  //int nchan=4096;
  //int ndat=12000;
  int nchan=1024;
  int ndat=327680;

  int nrep=1;

  if (argc>1)
    nchan=atoi(argv[1]);
  if (argc>2)
    ndat=atoi(argv[2]);
  if (argc>3)
    nrep=atoi(argv[3]);

  float **dat=matrix(nchan,ndat+nchan);
  float **dat2=matrix(nchan,ndat+nchan);
  if (1)
    for (int i=0;i<nchan;i++)
      dat[i][(int)(0.8317*i+160.2)]=1;
  else
    for (int i=0;i<nchan;i++)
      dat[i][ndat/2]=1;
  
#if 0
  write_mat(dat,nchan,ndat,"dat_starting.dat");
  dedisperse_kernel(dat,dat2,nchan,ndat);
  write_mat(dat2,nchan,ndat,"dat_1pass.dat");
  dedisperse_2pass(dat,dat2,nchan,ndat);
  write_mat(dat,nchan,ndat,"dat_2pass.dat");  
#endif



  double t1=omp_get_wtime();
  //dedisperse(dat,dat2,nchan,ndat);
  //dedisperse_blocked(dat,dat2,nchan,ndat);
  dedisperse_blocked_cached(dat,dat2,nchan,ndat);
  double t2=omp_get_wtime();
  printf("took %12.4f seconds.\n",t2-t1);
  int ichan,idat;
  find_peak(dat,nchan,ndat,&ichan,&idat);
  t1=omp_get_wtime();
  printf("took %12.4f seconds to find peak.\n",t1-t2);


  
  for (int i=0;i<10;i++) {
    
    t1=omp_get_wtime();
    for (int j=0;j<nrep;j++) {
      dedisperse_blocked_cached(dat,dat2,nchan,ndat);
      //dedisperse(dat,dat2,nchan,ndat);
    }
    t2=omp_get_wtime();
    double nops=get_npass(nchan)*(nchan+0.0)*(ndat+0.0)*(nrep+0.0);
    printf("took %12.6f seconds at rate %12.6f.\n",t2-t1,nops/(t2-t1)/1024/1024);

    //printf("took %12.4f seconds.\n",t2-t1);
  
  }
  
  //write_mat(dat,nchan,ndat,"dat_final1.dat");

  //write_mat(dat2,nchan,ndat,"dat_final2.dat");
}
