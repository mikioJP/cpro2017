//
//
//
// author: Kyohei Shimozato
// Usage example:
//    ./nn6 fc1.dat fc2.dat fc3.dat
//
// Input Params example
//    learning_rate:  0.005
//    epoc:           10rrrrrfv
//
//
//

#include "nn.h"
#include <time.h>

typedef struct {
  float A1[784*50];
  float b1[50];
  float A2[50*100];
  float b2[100];
  float A3[100*10];
  float b3[10];
} P;

void print(int m, int n, const float * x){
  int i,j;
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      printf("%.4f ",x[n*i+j]);
    }
    putchar('\n');
  }
}
void copy(int n,const float *x,float *y){
  int i;
  for(i=0;i<n;i++){
    y[i]=x[i];
  }
}
void fc(int m,int n,const float *x,const float *A,const float *b,float *y){
  //y=Ax+b
  int i,j;
  for(i=0;i<m;i++){
    y[i]=0.0;
    for(j=0;j<n;j++){
      y[i]+=A[i*n+j]*x[j];
    }
    y[i]+=b[i];
  }
}
void relu(int n,const float *x,float *y){
  //remove x<=0
  int i;
  for(i=0;i<n;i++){
    if(x[i]>0){
      y[i]=x[i];
    }else{
      y[i]=0;
    }
  }
}
void softmax(int n,const float *x,float *y){
  int i;
  float x_max=0.0;
  float exp_sum=0.0;

  for(i=0;i<n;i++){
    x_max=x[i]>x_max?x[i]:x_max;
  }
  for(i=0;i<n;i++){
    exp_sum+=exp(x[i]-x_max);
  }
  for(i=0;i<n;i++){
    y[i]=exp(x[i]-x_max)/exp_sum;
  }
}
int inference6( 
    const float *A1,const float *b1,
    const float *A2,const float *b2,
    const float *A3,const float *b3,
    const float *x,float *y){

  int i,ans;
  float max=0.0;

  float y_50[50];
  float y_100[100];

  fc(50,784,x,A1,b1,y_50);
  relu(50,y_50,y_50);
  fc(100,50,y_50,A2,b2,y_100);
  relu(100,y_100,y_100);
  fc(10,100,y_100,A3,b3,y);
  softmax(10,y,y);
  
  for(i=0;i<10;i++){
    if(y[i]>max){
      max=y[i];
      ans=i;
    }
  }
  return ans;
}
void softmaxwithloss_bwd(int n, const float *y,unsigned char t,float *dx){
  copy(n,y,dx);
  dx[t]-=1;
}
void relu_bwd(int n,const float *x,const float *dy,float *dx){
  int i;
  for(i=0;i<n;i++){
    if(x[i]>0){
      dx[i]=dy[i];
    }else{
      dx[i]=0;
    }
  }
}
void fc_bwd(int m,int n,const float *x,const float *dy,const float *A,float *dA,float *db,float *dx){
  int i,j;
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      dA[n*i+j]=dy[i]*x[j];
    }
  }

  copy(10,dy,db);

  for(i=0;i<n;i++){
    dx[i]=0;
    for(j=0;j<m;j++){
      dx[i]+=A[n*j+i]*dy[j];
    }
  }
}
void backward6(
    const P *p,
    const float *x,unsigned char t,
    P *dp
    ){

  float x_fc1     [784];
  float x_relu1   [50];
  float x_fc2     [50];
  float x_relu2   [100];
  float x_fc3     [100];

  float dx_fc1    [784];
  float dx_relu1  [50];
  float dx_fc2    [50];
  float dx_relu2  [100];
  float dx_fc3    [100];
  float dx_soft   [10];

  float y         [10];

  copy(784,x,x_fc1);
  fc(50,784,x_fc1,p->A1,p->b1,x_relu1);
  relu(50,x_relu1,x_fc2);
  fc(100,50,x_fc2,p->A2,p->b2,x_relu2);
  relu(100,x_relu2,x_fc3);
  fc(10,100,x_fc3,p->A3,p->b3,y);
  softmax(10,y,y);

  softmaxwithloss_bwd(10,y,t,dx_soft);
  fc_bwd(10,100,x_fc3,dx_soft,p->A3,dp->A3,dp->b3,dx_fc3);
  relu_bwd(100,x_relu2,dx_fc3,dx_relu2);
  fc_bwd(100,50,x_fc2,dx_relu2,p->A2,dp->A2,dp->b2,dx_fc2);
  relu_bwd(50,x_relu1,dx_fc2,dx_relu1);
  fc_bwd(50,784,x_fc1,dx_relu1,p->A1,dp->A1,dp->b1,dx_fc1);
}
void shuffle(int n, int *x){
  int i,j,x_i;
  for(i=0;i<n;i++){
    j=(int)(rand()*(float)n/(1.0+RAND_MAX));
    x_i=x[i]; //一時的に保存
    x[i]=x[j];
    x[j]=x_i;
  }
}
float loss(const float y, unsigned char t){
  return -1*t*log(y+0.0000001);
}
void add(int n,const float *x,float *o){
  int i;
  for(i=0;i<n;i++){
    o[i]+=x[i];
  }
}
void add6(const P *x_p,P *o_p){
  add(784*50 ,x_p->A1,o_p->A1);
  add(50     ,x_p->b1,o_p->b1);
  add(50*100 ,x_p->A2,o_p->A2);
  add(100    ,x_p->b2,o_p->b2);
  add(100*10 ,x_p->A3,o_p->A3);
  add(10     ,x_p->b3,o_p->b3);
}
void scale(int n,float x,float *o){
  int i;
  for(i=0;i<n;i++){
    o[i]*=x;
  }
}
void scale6(float x,P *o_p){
  scale(784*50 ,x,o_p->A1);
  scale(50     ,x,o_p->b1);
  scale(50*100 ,x,o_p->A2);
  scale(100    ,x,o_p->b2);
  scale(100*10 ,x,o_p->A3);
  scale(10     ,x,o_p->b3);
}
void init(int n,float x,float *o){
  int i;
  for(i=0;i<n;i++){
    o[i]=x;
  }
}
void init6(P *p){
  init(784*50 ,0,p->A1);
  init(50     ,0,p->b1);
  init(50*100 ,0,p->A2);
  init(100    ,0,p->b2);
  init(100*10 ,0,p->A3);
  init(10     ,0,p->b3);
}
void randinit(int n,float *o){
  int i;
  for(i=0;i<n;i++){
    o[i]=-1.0+rand()*2.0/(1.0+RAND_MAX);
;
  }
}
void randinit6(P *p){
  randinit(784*50 ,p->A1);
  randinit(50     ,p->b1);
  randinit(50*100 ,p->A2);
  randinit(100    ,p->b2);
  randinit(100*10 ,p->A3);
  randinit(10     ,p->b3);
}
void save(const char *filename,int m,int n,const float *A,const float *b){
  FILE *file=fopen(filename,"w");
  if(file==NULL){
    printf("Can't save %s.\n",filename);
  }else{
    fwrite(A,sizeof(float),m*n,file);
    fwrite(b,sizeof(float),m,file);
    fclose(file);
  }
}
void load(const char *filename,int m,int n,float *A, float *b){
  FILE *file=fopen(filename,"r");
  if(file==NULL){
    printf("can't load %s.\n",filename);
  }else{
    fread(A,sizeof(float),m*n,file);
    fread(b,sizeof(float),m,file);
    fclose(file);
    printf("Successfully loaded %s\n",filename);
  }
}
void test(const float *test_x,const unsigned char *test_y,
    int size,int test_count,
    const P *p,
    float *result,float *result_loss){

  int i,sum=0;
  float e=0.0;
  float y[10];

  for(i=0;i<test_count;i++){
    if(inference6(
          p->A1,p->b1,
          p->A2,p->b2,
          p->A3,p->b3,
          test_x+i*size,y) ==test_y[i]){
        sum++;
    }
      e+=loss(y[test_y[i]],1);
  }
  *result=sum*100.0/test_count;
  *result_loss=e/test_count;
}

int main(int argc,char *argv[]) {
  float * train_x = NULL;
  unsigned char * train_y = NULL;
  int train_count = -1;

  float * test_x = NULL;
  unsigned char * test_y = NULL;
  int test_count = -1;

  int width = -1;
  int height = -1;

  load_mnist(&train_x, &train_y, &train_count,
             &test_x, &test_y, &test_count,
             &width, &height);
  
  //rand関数初期化
  srand((unsigned int)time(NULL));
  rand();rand();rand();rand();rand();

  int i,j,k,epoc,batch_size=100,batch_count;
  float learning_rate,result,result_loss;
  int index[train_count];

  P *p      =(P*)malloc(sizeof(P));
  P *dp     =(P*)malloc(sizeof(P));
  P *dp_ave =(P*)malloc(sizeof(P));

  printf("learning_rate:"); scanf("%f",&learning_rate);
  printf("epoc:"); scanf("%d",&epoc);
  batch_count=train_count/batch_size;

  for(i=0;i<train_count;i++){
    index[i]=i;
  }
  randinit6(p);

  if(argc>0){
    load(argv[1],50,784,p->A1,p->b1);
    load(argv[2],100,50,p->A2,p->b2);
    load(argv[3],10,100,p->A3,p->b3);
  }

  for(i=0;i<epoc;i++){
    shuffle(train_count,index);
    for(j=0;j<batch_count;j++){
      init6(dp_ave);
      for(k=0;k<batch_size;k++){
        backward6(p,
            train_x+width*height*index[j*batch_size+k],
            train_y[index[j*batch_size+k]],
            dp);
        add6(dp,dp_ave);
      }
      scale6(-1*learning_rate/batch_size,dp_ave);
      add6(dp_ave,p);
    }
    test(test_x,test_y,width*height,test_count,
        p,&result,&result_loss);
    printf("%2d回目:%.2f%% loss:%.3f\n",i+1,result,result_loss);
  }

  if(argc>0){
    save(argv[1],50,784,p->A1,p->b1);
    save(argv[2],100,50,p->A2,p->b2);
    save(argv[3],10,100,p->A3,p->b3);
  }

  free(p);
  free(dp);
  free(dp_ave);

  return 0;
}
