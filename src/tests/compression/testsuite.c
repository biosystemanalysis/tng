/* tng compression routines */

/* Only modify testsuite.c
 *Then* run testsuite.sh to perform the test.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tng_compress.h>
#include <warnmalloc.h>
#include TESTPARAM

#define FUDGE 1.1 /* 10% off target precision is acceptable */

static void keepinbox(int *val)
{
  while (val[0]>INTMAX1)
    val[0]-=(INTMAX1-INTMIN1+1);
  while (val[0]<INTMIN1)
    val[0]+=(INTMAX1-INTMIN1+1);
  while (val[1]>INTMAX2)
    val[1]-=(INTMAX2-INTMIN2+1);
  while (val[1]<INTMIN2)
    val[1]+=(INTMAX2-INTMIN2+1);
  while (val[2]>INTMAX3)
    val[2]-=(INTMAX3-INTMIN3+1);
  while (val[2]<INTMIN3)
    val[2]+=(INTMAX3-INTMIN3+1);
}

static int intsintable[128]={
0 , 3215 , 6423 , 9615 , 12785 , 15923 , 19023 , 22078 , 
25079 , 28019 , 30892 , 33691 , 36409 , 39039 , 41574 , 44010 , 
46340 , 48558 , 50659 , 52638 , 54490 , 56211 , 57796 , 59242 , 
60546 , 61704 , 62713 , 63570 , 64275 , 64825 , 65219 , 65456 , 
65535 , 65456 , 65219 , 64825 , 64275 , 63570 , 62713 , 61704 , 
60546 , 59242 , 57796 , 56211 , 54490 , 52638 , 50659 , 48558 , 
46340 , 44010 , 41574 , 39039 , 36409 , 33691 , 30892 , 28019 , 
25079 , 22078 , 19023 , 15923 , 12785 , 9615 , 6423 , 3215 , 
0 , -3215 , -6423 , -9615 , -12785 , -15923 , -19023 , -22078 , 
-25079 , -28019 , -30892 , -33691 , -36409 , -39039 , -41574 , -44010 , 
-46340 , -48558 , -50659 , -52638 , -54490 , -56211 , -57796 , -59242 , 
-60546 , -61704 , -62713 , -63570 , -64275 , -64825 , -65219 , -65456 , 
-65535 , -65456 , -65219 , -64825 , -64275 , -63570 , -62713 , -61704 , 
-60546 , -59242 , -57796 , -56211 , -54490 , -52638 , -50659 , -48558 , 
-46340 , -44010 , -41574 , -39039 , -36409 , -33691 , -30892 , -28019 , 
-25079 , -22078 , -19023 , -15923 , -12785 , -9615 , -6423 , -3215 , 
};

static int intsin(int i)
{
  int sign=1;
  if (i<0)
    {
      i=0;
      sign=-1;
    }
  return sign*intsintable[i%128];
}

static int intcos(int i)
{
  if (i<0)
    i=0;
  return intsin(i+32);
}

static void molecule(int *target, 
		     int *base,
		     int length,
		     int scale, int *direction,
		     int flip,
		     int iframe)
{
  int i;
  for (i=0; i<length; i++)
    {
      int ifl=i;
      if ((i==0) && (flip) && (length>1))
	ifl=1;
      else if ((i==1) && (flip) && (length>1))
	ifl=0;
      target[ifl*3]=base[0]+(intsin((i+iframe)*direction[0])*scale)/256;
      target[ifl*3+1]=base[1]+(intcos((i+iframe)*direction[1])*scale)/256;
      target[ifl*3+2]=base[2]+(intcos((i+iframe)*direction[2])*scale)/256;
      keepinbox(target+ifl*3);
    }
}

#ifndef FRAMESCALE
#define FRAMESCALE 1
#endif

static void genibox(int *intbox, int iframe)
{
  int molecule_length=1;
  int molpos[3];
  int direction[3]={1,1,1};
  int scale=1;
  int flip=0;
  int i=0;
  molpos[0]=intsin(iframe*FRAMESCALE)/32;
  molpos[1]=1+intcos(iframe*FRAMESCALE)/32;
  molpos[2]=2+intsin(iframe*FRAMESCALE)/16;
  keepinbox(molpos);
  while (i<NATOMS)
    {
      int this_mol_length=molecule_length;
      int dir;
#ifdef REGULAR
      this_mol_length=4;
      flip=0;
      scale=1;
#endif
      if (i+this_mol_length>NATOMS)
	this_mol_length=NATOMS-i;
      /* We must test the large rle as well. This requires special
	 sequencies to get triggered. So insert these from time to
	 time */
#ifndef REGULAR
      if ((i%10)==0)
	{
	  int j;
	  intbox[i*3]=molpos[0];
	  intbox[i*3+1]=molpos[1];
	  intbox[i*3+2]=molpos[2];
	  for (j=1; j<this_mol_length; j++)
	    {
	      intbox[(i+j)*3]=intbox[(i+j-1)*3]+(INTMAX1-INTMIN1+1)/5;
	      intbox[(i+j)*3+1]=intbox[(i+j-1)*3+1]+(INTMAX2-INTMIN2+1)/5;
	      intbox[(i+j)*3+2]=intbox[(i+j-1)*3+2]+(INTMAX3-INTMIN3+1)/5;
	      keepinbox(intbox+(i+j)*3);
	    }
	}
      else
#endif
	molecule(intbox+i*3,molpos,this_mol_length,scale,direction,flip,iframe*FRAMESCALE);
      i+=this_mol_length;
      dir=1;
      if (intsin(i*3)<0)
	dir=-1;
      molpos[0]+=dir*(INTMAX1-INTMIN1+1)/20;
      dir=1;
      if (intsin(i*5)<0)
	dir=-1;
      molpos[1]+=dir*(INTMAX2-INTMIN2+1)/20;
      dir=1;
      if (intsin(i*7)<0)
	dir=-1;
      molpos[2]+=dir*(INTMAX3-INTMIN3+1)/20;
      keepinbox(molpos);
      
      direction[0]=((direction[0]+1)%7)+1;
      direction[1]=((direction[1]+1)%3)+1;
      direction[2]=((direction[2]+1)%6)+1;

      scale++;
      if (scale>5)
	scale=1;

      molecule_length++;
      if (molecule_length>30)
	molecule_length=1;
      if (i%9)
	flip=1-flip;
    }
}

static void genivelbox(int *intvelbox, int iframe)
{
  int i;
  for (i=0; i<NATOMS; i++)
    {
#ifdef VELINTMUL
      intvelbox[i*3]=((intsin((i+iframe*FRAMESCALE)*3))/10)*VELINTMUL+i;
      intvelbox[i*3+1]=1+((intcos((i+iframe*FRAMESCALE)*5))/10)*VELINTMUL+i;
      intvelbox[i*3+2]=2+((intsin((i+iframe*FRAMESCALE)*7)+intcos((i+iframe*FRAMESCALE)*9))/20)*VELINTMUL+i;
#else
      intvelbox[i*3]=((intsin((i+iframe*FRAMESCALE)*3))/10);
      intvelbox[i*3+1]=1+((intcos((i+iframe*FRAMESCALE)*5))/10);
      intvelbox[i*3+2]=2+((intsin((i+iframe*FRAMESCALE)*7)+intcos((i+iframe*FRAMESCALE)*9))/20);
#endif
    }
}

#ifndef STRIDE1
#define STRIDE1 3
#endif

#ifndef STRIDE2
#define STRIDE2 3
#endif

#ifndef GENPRECISION
#define GENPRECISION PRECISION
#endif

#ifndef GENVELPRECISION
#define GENVELPRECISION VELPRECISION
#endif

static void realbox(int *intbox, double *realbox, int stride)
{
  int i,j;
  for (i=0; i<NATOMS; i++)
    {
      for (j=0; j<3; j++)
	realbox[i*stride+j]=(double)(intbox[i*3+j]*GENPRECISION*SCALE);
      for (j=3; j<stride; j++)
	realbox[i*stride+j]=0.;
    }
}

static void realvelbox(int *intbox, double *realbox, int stride)
{
  int i,j;
  for (i=0; i<NATOMS; i++)
    {
      for (j=0; j<3; j++)
	realbox[i*stride+j]=(double)(intbox[i*3+j]*GENVELPRECISION*SCALE);
      for (j=3; j<stride; j++)
	realbox[i*stride+j]=0.;
    }
}

static int equalarr(double *arr1, double *arr2, double prec, int len, int itemlen, int stride1, int stride2)
{
  double maxdiff=0.;
  int i,j;
  for (i=0; i<len; i++)
    {
      for (j=0; j<itemlen; j++)
	if (fabs(arr1[i*stride1+j]-arr2[i*stride2+j])>maxdiff)
	  maxdiff=(double)fabs(arr1[i*stride1+j]-arr2[i*stride2+j]);
    }
#if 0
  for (i=0; i<len; i++)
    {
      for (j=0; j<itemlen; j++)
	printf("%d %d: %g %g\n",i,j,arr1[i*stride1+j],arr2[i*stride2+j]);
    }
#endif
#if 0
  fprintf(stderr,"Error is %g. Acceptable error is %g.\n",maxdiff,prec*0.5*FUDGE);
#endif
  if (maxdiff>prec*0.5*FUDGE)
    {
      return 0;
    }
  else
    return 1;
}

struct tng_file
{
  FILE *f;
  int natoms;
  int chunky;
  double precision;
  double velprecision;
  int initial_coding;
  int initial_coding_parameter;
  int coding;
  int coding_parameter;
  int initial_velcoding;
  int initial_velcoding_parameter;
  int velcoding;
  int velcoding_parameter;
  int speed;
  int nframes;
  int nframes_delivered;
  int writevel;
  double *pos;
  double *vel;
};

static size_t fwrite_int_le(int *x,FILE *f)
{
  unsigned char c[4];
  unsigned int i=(unsigned int)*x;
  c[0]=(unsigned char)(i&0xFFU);
  c[1]=(unsigned char)((i>>8)&0xFFU);
  c[2]=(unsigned char)((i>>16)&0xFFU);
  c[3]=(unsigned char)((i>>24)&0xFFU);
  return fwrite(c,1,4,f);
}

static size_t fread_int_le(int *x,FILE *f)
{
  unsigned char c[4];
  unsigned int i;
  size_t n=fread(c,1,4,f);
  if (n)
    {
      i=(((unsigned int)c[3])<<24)|(((unsigned int)c[2])<<16)|(((unsigned int)c[1])<<8)|((unsigned int)c[0]);
      *x=(int)i;
    }
  return n;
}

static struct tng_file *open_tng_file_write(char *filename,
					    int natoms,int chunky,
					    double precision,
					    int writevel,
					    double velprecision,
					    int initial_coding,
					    int initial_coding_parameter,
					    int coding,
					    int coding_parameter,
					    int initial_velcoding,
					    int initial_velcoding_parameter,
					    int velcoding,
					    int velcoding_parameter,
					    int speed)
{
  struct tng_file *tng_file=malloc(sizeof *tng_file);
  tng_file->pos=NULL;
  tng_file->vel=NULL;
  tng_file->nframes=0;
  tng_file->chunky=chunky;
  tng_file->precision=precision;
  tng_file->natoms=natoms;
  tng_file->writevel=writevel;
  tng_file->velprecision=velprecision;
  tng_file->initial_coding=initial_coding;
  tng_file->initial_coding_parameter=initial_coding_parameter;
  tng_file->coding=coding;
  tng_file->coding_parameter=coding_parameter;
  tng_file->initial_velcoding=initial_velcoding;
  tng_file->initial_velcoding_parameter=initial_velcoding_parameter;
  tng_file->velcoding=velcoding;
  tng_file->velcoding_parameter=velcoding_parameter;
  tng_file->speed=speed;
  tng_file->pos=malloc(natoms*chunky*3*sizeof *tng_file->pos);
  tng_file->f=fopen(filename,"wb");
  if (writevel)
    tng_file->vel=malloc(natoms*chunky*3*sizeof *tng_file->vel);
  fwrite_int_le(&natoms,tng_file->f);
  return tng_file;
}

static void flush_tng_frames(struct tng_file *tng_file)
{
  int algo[4];
  char *buf;
  int nitems;
  fwrite_int_le(&tng_file->nframes,tng_file->f);
  algo[0]=tng_file->initial_coding;
  algo[1]=tng_file->initial_coding_parameter;
  algo[2]=tng_file->coding;
  algo[3]=tng_file->coding_parameter;
  buf=tng_compress_pos(tng_file->pos,
		       tng_file->natoms,
		       tng_file->nframes,
		       tng_file->precision,
		       tng_file->speed,algo,&nitems);
  tng_file->initial_coding=algo[0];
  tng_file->initial_coding_parameter=algo[1];
  tng_file->coding=algo[2];
  tng_file->coding_parameter=algo[3];
  fwrite_int_le(&nitems,tng_file->f);
  fwrite(buf,1,nitems,tng_file->f);
  free(buf);
  if (tng_file->writevel)
    {
      algo[0]=tng_file->initial_velcoding;
      algo[1]=tng_file->initial_velcoding_parameter;
      algo[2]=tng_file->velcoding;
      algo[3]=tng_file->velcoding_parameter;
      buf=tng_compress_vel(tng_file->vel,
			   tng_file->natoms,
			   tng_file->nframes,
			   tng_file->velprecision,
			   tng_file->speed,algo,&nitems);
      tng_file->initial_velcoding=algo[0];
      tng_file->initial_velcoding_parameter=algo[1];
      tng_file->velcoding=algo[2];
      tng_file->velcoding_parameter=algo[3];
      fwrite_int_le(&nitems,tng_file->f);
      fwrite(buf,1,nitems,tng_file->f);
      free(buf);
    }
  tng_file->nframes=0;
}

static void write_tng_file(struct tng_file *tng_file,
			   double *pos,double *vel)
{
  memcpy(tng_file->pos+tng_file->nframes*tng_file->natoms*3,pos,tng_file->natoms*3*sizeof *tng_file->pos);
  if (tng_file->writevel)
    memcpy(tng_file->vel+tng_file->nframes*tng_file->natoms*3,vel,tng_file->natoms*3*sizeof *tng_file->vel);
  tng_file->nframes++;
  if (tng_file->nframes==tng_file->chunky)
    flush_tng_frames(tng_file);
}

static void close_tng_file_write(struct tng_file *tng_file)
{
  if (tng_file->nframes)
    flush_tng_frames(tng_file);
  fclose(tng_file->f);
  free(tng_file->pos);
  free(tng_file->vel);
  free(tng_file);
}

static struct tng_file *open_tng_file_read(char *filename, int writevel)
{
  struct tng_file *tng_file=malloc(sizeof *tng_file);
  tng_file->pos=NULL;
  tng_file->vel=NULL;
  tng_file->f=fopen(filename,"rb");
  tng_file->nframes=0;
  tng_file->nframes_delivered=0;
  tng_file->writevel=writevel;
  fread_int_le(&tng_file->natoms,tng_file->f);
  return tng_file;
}

static int read_tng_file(struct tng_file *tng_file,
			 double *pos,
			 double *vel)
{
  if (tng_file->nframes==tng_file->nframes_delivered)
    {
      int nitems;
      char *buf;
      free(tng_file->pos);
      free(tng_file->vel);
      if (!fread_int_le(&tng_file->nframes,tng_file->f))
	return 1;
      if (!fread_int_le(&nitems,tng_file->f))
	return 1;
      buf=malloc(nitems);
      if (!fread(buf,1,nitems,tng_file->f))
	return 1;
      tng_file->pos=malloc(tng_file->natoms*tng_file->nframes*3*sizeof *tng_file->pos);
      if (tng_file->writevel)
	tng_file->vel=malloc(tng_file->natoms*tng_file->nframes*3*sizeof *tng_file->vel);
#if 0
      {
	int natoms, nframes, algo[4];
	double precision;
	int ivel;
	char *initial_coding, *coding;
	tng_compress_inquire(buf,&ivel,&natoms,&nframes,&precision,algo);
	initial_coding=tng_compress_initial_pos_algo(algo);
	coding=tng_compress_pos_algo(algo);
	printf("ivel=%d natoms=%d nframes=%d precision=%g initial pos=%s pos=%s\n",ivel,natoms,nframes,precision,initial_coding,coding);
      }
#endif
      tng_compress_uncompress(buf,tng_file->pos);
      free(buf);
      if (tng_file->writevel)
	{
	  if (!fread_int_le(&nitems,tng_file->f))
	    return 1;
	  buf=malloc(nitems);
	  if (!fread(buf,1,nitems,tng_file->f))
	    return 1;
#if 0      
	  {
	    int natoms, nframes, algo[4];
	    double precision;
	    int ivel;
	    char *initial_coding, *coding;
	    tng_compress_inquire(buf,&ivel,&natoms,&nframes,&precision,algo);
	    initial_coding=tng_compress_initial_vel_algo(algo);
	    coding=tng_compress_vel_algo(algo);
	    printf("ivel=%d natoms=%d nframes=%d precision=%g initial vel=%s vel=%s\n",ivel,natoms,nframes,precision,initial_coding,coding);
	  }
#endif
	  tng_compress_uncompress(buf,tng_file->vel);
	  free(buf);
	}
      tng_file->nframes_delivered=0;
    }
  memcpy(pos,tng_file->pos+tng_file->nframes_delivered*tng_file->natoms*3,tng_file->natoms*3*sizeof *pos);
  if (tng_file->writevel)
    memcpy(vel,tng_file->vel+tng_file->nframes_delivered*tng_file->natoms*3,tng_file->natoms*3*sizeof *vel);
  tng_file->nframes_delivered++;
  return 0;
}

static void close_tng_file_read(struct tng_file *tng_file)
{
  free(tng_file->vel);
  free(tng_file->pos);
  fclose(tng_file->f);
  free(tng_file);
}

		       

#ifndef EXPECTED_FILESIZE
#define EXPECTED_FILESIZE 1
#endif

#ifndef INITIALVELCODING
#define INITIALVELCODING -1
#endif
#ifndef INITIALVELCODINGPARAMETER
#define INITIALVELCODINGPARAMETER -1
#endif

#ifndef SPEED
#define SPEED 5
#endif

/* Return value 1 means file error.
   Return value 4 means coding error in coordinates.
   Return value 5 means coding error in velocities.
   Return value 9 means filesize seems too off.

   Return value 100+ means test specific error.
 */
static int algotest()
{
  int i;
  int *intbox=warnmalloc(NATOMS*3*sizeof *intbox);
  int *intvelbox=warnmalloc(NATOMS*3*sizeof *intvelbox);
  double *box1=warnmalloc(NATOMS*STRIDE1*sizeof *box1);
  double *velbox1=warnmalloc(NATOMS*STRIDE1*sizeof *velbox1);
  double time1, lambda1;
  double H1[9];
  int startframe=0;
  int endframe=NFRAMES;
#ifdef GEN
  FILE *file;
  double filesize;
#else
  int i2;
  int readreturn;
  double H2[9];
  double time2, lambda2;
  double *box2=warnmalloc(NATOMS*STRIDE2*sizeof *box2);
  double *velbox2=warnmalloc(NATOMS*STRIDE2*sizeof *velbox2);
#endif
#ifdef GEN
  void *dumpfile=open_tng_file_write(FILENAME,NATOMS,CHUNKY,
				     PRECISION,WRITEVEL,VELPRECISION,
				     INITIALCODING,
				     INITIALCODINGPARAMETER,CODING,CODINGPARAMETER,
				     INITIALVELCODING,INITIALVELCODINGPARAMETER,
				     VELCODING,VELCODINGPARAMETER,SPEED);
#else
  void *dumpfile=open_tng_file_read(FILENAME,WRITEVEL);
#endif
  if (!dumpfile)
    return 1;
  for (i=0; i<9; i++)
    H1[i]=0.;
  H1[0]=INTMAX1*PRECISION*SCALE;
  H1[4]=INTMAX2*PRECISION*SCALE;
  H1[8]=INTMAX3*PRECISION*SCALE;
  for (i=startframe; i<endframe; i++)
    {
      genibox(intbox,i);
      realbox(intbox,box1,STRIDE1);
#if WRITEVEL
      genivelbox(intvelbox,i);
      realvelbox(intvelbox,velbox1,STRIDE1);
#endif
      time1=(double)i;
      lambda1=(double)(i+100);
#ifdef GEN
      write_tng_file(dumpfile,box1,velbox1);
#else
      readreturn=read_tng_file(dumpfile,box2,velbox2);
      if (readreturn==1) /* general read error  */
	return 1;
#endif
#ifndef GEN
      /* Check for equality of boxes. */
      if (!equalarr(box1,box2,(double)PRECISION,NATOMS,3,STRIDE1,STRIDE2))
	return 4;
#if WRITEVEL
      if (!equalarr(velbox1,velbox2,(double)VELPRECISION,NATOMS,3,STRIDE1,STRIDE2))
	return 5;
#endif
#endif
    }
#ifdef GEN
  close_tng_file_write(dumpfile);
#else
  close_tng_file_read(dumpfile);
#endif
#ifdef GEN
  /* Check against expected filesize for this test. */
  if (!(file=fopen(FILENAME,"rb")))
    {
      fprintf(stderr,"ERROR: Cannot open file "FILENAME"\n");
      exit(EXIT_FAILURE);
    }
  filesize=0;
  while(1)
    {
      char b;
      if (!fread(&b,1,1,file))
	break;
      filesize++;
    }
  fclose(file);
  if (filesize>0)
    {
      if ((fabs(filesize-EXPECTED_FILESIZE)/EXPECTED_FILESIZE)>0.05)
	return 9;
    }
#endif
  return 0;
}

int main()
{
  int testval;
  if (sizeof(int)<4)
    {
      fprintf(stderr,"ERROR: sizeof(int) is too small: %d<4\n",(int)sizeof(int));
      exit(EXIT_FAILURE);
    }
#ifdef GEN
  printf("Tng compress testsuite generating test: %s\n",TESTNAME);
#else
  printf("Tng compress testsuite running test: %s\n",TESTNAME);
#endif
  testval=algotest();
  if (testval==0)
    printf("Passed.\n");
  else if (testval==1)
    {
      printf("ERROR: File error.\n");
      exit(EXIT_FAILURE);
    }
  else if (testval==4)
    {
      printf("ERROR: Read coding error in coordinates.\n");
      exit(EXIT_FAILURE);
    }
  else if (testval==5)
    {
      printf("ERROR: Read coding error in velocities.\n");
      exit(EXIT_FAILURE);
    }
  else if (testval==9)
    {
      printf("ERROR: Generated filesize differs too much.\n");
      exit(EXIT_FAILURE);
    }
  else
    {
      printf("ERROR: Unknown error.\n");
      exit(EXIT_FAILURE);
    }
  return 0;
}