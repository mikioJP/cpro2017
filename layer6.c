#include "nn.h"
#include <time.h>

//
void print(int n, int m, const float *x){
    int j, i;
    for (j = 0; j < n ; ++j) {
        for (i = 0; i < m ; ++i) {
            printf("%.4lf ", x[i + n * j]);
        }
        printf("\n");
    }

    printf("\n");
}

//配列を別のものにコピーする
void copy(int n, const float *x, float *y){
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

//fc層
void fc(int m, int n, const float * x, const float * A, const float * b, float * y){
    int i,j,p;

    for(i = 0; i < m ; i++){
        y[i] = 0.0;
        for(j = 0; j < n; j++){
            p = n * i + j;
            y[i] += A[p] * x[j] ;
        }
        y[i] += b[i];
    }
}

//Relu層
void relu (int n, const float *x, float *y){
    int i;
    for (i = 0; i < n; ++i) {
        if (x[i] > 0){
            y[i] = x[i];
        }else{
            y[i]=0;
        }
    }
}

//softmax層
void softmax(int n, const float *x, float *y) {
    int i;
    float max = 0;
    float sum = 0.0;

    for(i = 0; i < n; i++) {
        if(max < x[i])
            max = x[i];
    }

    for(i = 0; i < n; i++) {
        sum += exp(x[i] - max);
    }

    for(i = 0; i < n; i++) {
        y[i] = exp(x[i] - max) / sum;
    }
}

//3層での推論
/*
int inference3(const float *A, const float *b, const float *x){
    float *y = malloc(sizeof(float)*10);
    int i;

    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);

    float max = 0;
    int index = 0;

    for (i = 0; i < 10 ; ++i) {
        if (max < y[i]){
            max = y[i];
            index = i;
        }
    }
    return  index;
}
*/
//６層での推論
int inference6 (const float *A1,const float *b1,
                const float *A2,const float *b2,
                const float *A3,const float *b3,
                const float *x,float *y){


    float *y1 =malloc(sizeof(float)*50);
    float *y2 =malloc(sizeof(float)*100);

    int i;

    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2,A3, b3, y);
    softmax(10, y, y);

    float max = 0.0;
    int index = 0;

    for (i = 0; i < 10 ; ++i) {
        if (max < y[i]){
            max = y[i];
            index = i;
        }
    }
    free(y1);free(y2);
    return  index;
}

//逆伝播のsoftmax
void softMaxWithLoss_bwd(int n, const float *y, unsigned char t, float * dx) {

    float correctAns[n];
    int i,j;
    for (j = 0; j < n; ++j) {
        if (j == t){
            correctAns[j] = 1;
        }else{
            correctAns[j] = 0;
        }
    }

    for (i = 0; i < n; ++i) {
        dx[i] =  y[i] - correctAns[i];
    }
}

//逆伝播のrelu
void relu_bwd(int n, const float *x, const float *dy, float *dx ){
    int i;
    for (i = 0; i < n; ++i) {
        if (x[i] > 0){
            dx[i] = dy[i];
        }else{
            dx[i] = 0;
        }
    }

}

//逆伝播のfc
void fc_bwd(int m,int n,const float *x,const float *dy,const float *A,float *dA,float *db,float *dx){
    int i,j;
    for(i = 0 ; i < m ; i++){
        for(j = 0 ; j < n ; j++){
            dA[n * i + j] = dy[i] * x[j];
        }
    }

    copy(m,dy,db);

    for( i = 0;i < n; i++){
        dx[i]=0;
        for(j = 0; j < m ; j++){
            dx[i] += A[n * j + i] * dy[j];
        }
    }
}

void backward3(const float *A, const float *b, const float *x, unsigned char t, float *y, float *dA, float *db ) {

    //float u = 0.01;


    float fcTemp[784];
    float ReLUTemp[10];
    float gradX[7840];

    //順伝播
    copy(784, x, fcTemp);
    fc(10, 784, x, A, b, y);
    copy(10, y, ReLUTemp);
    relu(10, y, y);
    softmax(10, y, y);

//逆伝播
    softMaxWithLoss_bwd(10, y, t, y);
    relu_bwd(10, ReLUTemp, y, y);
    fc_bwd(10, 784, fcTemp, y, A, dA, db, gradX);



    /*for(i=0; i<10*784; i++){
        A[i] = A[i] - u * dA[i];
    }
    for(i=0; i<10; i++){
        b[i] = b[i] - u * db[i];
    }*/


}

//softmax_bwd,relu_bwd,fc_bwdを使った6層の逆伝播
void backward6(const float *A1, const float *b1,
               const float *A2, const float *b2,
               const float *A3, const float *b3,
                float *x, unsigned char t,
                float *dA1,  float * db1,
                float *dA2,  float * db2,
                float *dA3,  float * db3){

    float fcTemp1[784];
    float ReLUTemp1[50];
    float fcTemp2[50];
    float ReLUTemp2[100];
    float fcTemp3[100];

    float dx_fc1[784];
    float dx_relu1[50];
    float dx_fc2[50];
    float dx_relu2[100];
    float dx_fc3[100];
    float dx_softMaxWithLoss[10];

    float y[10];

    //順伝播
    copy(784, x, fcTemp1);
    fc(50, 784, fcTemp1, A1, b1, ReLUTemp1);
    relu(50, ReLUTemp1, fcTemp2);
    fc(100, 50, fcTemp2, A2, b2, ReLUTemp2);
    relu(100, ReLUTemp2, fcTemp3);
    fc(10, 100, fcTemp3, A3, b3, y);
    softmax(10, y, y);

    //逆伝播
    softMaxWithLoss_bwd(10, y, t, dx_softMaxWithLoss);
    fc_bwd(10, 100, fcTemp3, dx_softMaxWithLoss, A3, dA3, db3, dx_fc3);
    relu_bwd(100, ReLUTemp2, dx_fc3, dx_relu2);
    fc_bwd(100, 50, fcTemp2, dx_relu2, A2, dA2, db2, dx_fc2);
    relu_bwd(50, ReLUTemp1, dx_fc2, dx_relu1);
    fc_bwd(50, 784, fcTemp1, dx_relu1, A1, dA1, db1, dx_fc1);

}

//indexを乱数でシャッフルする用
//参考にするだけにしてください(by mikioJP)
void shuffle(int n, int *x){
    int i;
    for (i = 0; i < n ; ++i) {
        //0からn-1までの乱数生成
        int num= (int)(rand() * (float) n / (1.0+RAND_MAX));
        //入れ替え
        int temp = x[num];
        x[num] = x[i];
        x[i] = temp;
    }
}

//交差エントロピー
//参考にするだけにしてください(by mikioJP)
float loss(const float y, unsigned char t){
    return (float) (-1 * t * log(y + 0.0000001));
}

//和計算
//参考にするだけにしてください(by mikioJP)
void add(int n,  const float *x, float *o){
    int i;
    for (i = 0; i < n; ++i) {
        o[i] += x[i];
    }
}

//定数倍
//参考にするだけにしてください(by mikioJP)
void scale(int n, float x, float *o){
    int i;
    for (i = 0; i < n; ++i) {
        o[i] *= x;
    }
}

//任意のfloatで初期化
//参考にするだけにしてください(by mikioJP)
void init (int n, float x, float *o){
    int i;
    for (i = 0; i < n; ++i) {
        o[i] = x;
    }
}

//乱数で初期化
/*
void rand_init(int n, float *o){

    srand((unsigned)time(NULL));
    int i;
    for (i = 0; i < n; ++i) {
        o[i] =  (float)rand() / (float)RAND_MAX * 2 -1;
    }

}*/
//参考にするだけにしてください(by mikioJP)


//ボックスミューラー法で正規分布の乱数を生成
//今見るとここが一番ダメなので参考にするだけにしてください(by mikioJP)

void muller_init(int n, float *o){
    srand((unsigned)time(NULL));
    double r1, r2;
    int i;

    for (i = 0; i < n; ++i) {
        r1 =  (double)rand() / (double)RAND_MAX;
        r2 =  (double)rand() / (double)RAND_MAX;
        o[i] = sqrt(-2*log(r1))*cos(2*M_PI*r2);
    }
}

//学習係数を保存
//参考にするだけにしてください(by mikioJP)
void save(const char *filename, int n, int m, const float *A,const float *b){

    FILE *file;
    file = fopen(filename, "w");

    if(file == NULL){
        printf("save error %s!.\n", filename);
    }else{
        fwrite(A, sizeof(float) ,m*n, file);
        fwrite(b, sizeof(float), n, file);
        fclose(file);
    }
}

//学習係数を読み込み
//参考にするだけにしてください(by mikioJP)
void load(const char *filename,int n,int m,float *A, float *b){

    FILE *file;
    file = fopen(filename,"r");

    if(file==NULL){
        printf(" load error %s!.\n",filename);
    }else{
        fread(A,sizeof(float),n*m,file);
        fread(b,sizeof(float),n,file);
        fclose(file);
        
        printf("Loaded %s!\n",filename);
    }
}



int main() {
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);

    // これ以降，３層NNの係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x[0]～train_x[train_count-1], train_y[0]～train_x[train_count-1],
    // テストデータ test_x[0]～test_x[test_count-1], test_y[0]～test_y[test_count-1],
    // を使用することができる．頑張って

    //randの初期化
    srand(time(NULL));


    //メモリ空間確保
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);

    float *dA1 = malloc(sizeof(float) * 784 * 50);
    float *db1 = malloc(sizeof(float) * 50);
    float *dA2 = malloc(sizeof(float) * 50 * 100);
    float *db2 = malloc(sizeof(float) * 100);
    float *dA3 = malloc(sizeof(float) * 100 * 10);
    float *db3 = malloc(sizeof(float) * 10);


    float *AVEdA1= malloc(sizeof(float) * 784 * 50);
    float *AVEdb1 = malloc(sizeof(float) * 50);
    float *AVEdA2 = malloc(sizeof(float) * 50 * 100);
    float *AVEdb2 = malloc(sizeof(float) * 100);
    float *AVEdA3 = malloc(sizeof(float) * 100 * 10);
    float *AVEdb3 = malloc(sizeof(float) * 10);

    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);


    //係数を[-1:1]で初期化
    muller_init(784 * 50, A1);
    muller_init(50, b1);
    muller_init(50 * 100 , A2);
    muller_init(100, b2);
    muller_init(100 * 10, A3);
    muller_init(10, b3);

    //ミニバッチサイズ100からN =600
    int batchSize =100;
    int N =train_count;
    float  u = 0.01;
    int index[train_count];
    int correctCount = 0;
    float LossValue;
    int epocTimes = 10;
    int i,j,k,l;

    //index生成
    for (i = 0; i < train_count; ++i) {
        index[i] = i;
    }


    //以下ミニバッチ学習
    //コピペ対策

    for (l = 0; l < epocTimes; ++l) {
        //index並び替え
        shuffle(train_count, index);

        for (k = 0; k < N/batchSize; ++k) {

            //平均値初期化
            init(784*50, 0.0, AVEdA1);
            init(50, 0.0, AVEdb1);
            init(50*100, 0.0, AVEdA2);
            init(100, 0.0, AVEdb2);
            init(100*10, 0.0, AVEdA3);
            init(10, 0.0, AVEdb3);



            for (j = 0; j < batchSize ; ++j) {

                backward6(A1, b1,
                          A2,b2,
                          A3, b3,
                          train_x + width*height*index[j+k*batchSize], train_y[index[j + k*batchSize]],
                          dA1, db1,
                          dA2, db2,
                          dA3, db3);


                add(784*50, dA1, AVEdA1);
                add(50, db1, AVEdb1);
                add(50*100, dA2, AVEdA2);
                add(100, db2, AVEdb2);
                add(100*10, dA3, AVEdA3);
                add(10, db3, AVEdb3);


            }

            //平均化と学習倍率をかける
            scale(784*50, -u/batchSize, AVEdA1);
            scale(50, -u/batchSize, AVEdb1);
            scale(50*100, -u/batchSize, AVEdA2);
            scale(100, -u/batchSize, AVEdb2);
            scale(100*10, -u/batchSize, AVEdA3);
            scale(10, -u/batchSize, AVEdb3);

            //反映
            add(784*50, AVEdA1, A1);
            add(50, AVEdb1, b1);
            add(50*100, AVEdA2, A2);
            add(100, AVEdb2, b2);
            add(100*10, AVEdA3, A3);
            add(10, AVEdb3, b3);

        }

        //精度評価
        correctCount = 0;
        LossValue = 0;

        for (i = 0; i < test_count; ++i) {
            if (inference6(A1,b1,
                           A2,b2,
                           A3,b3,
                           test_x+width*height*i,y1) == test_y[i]) {
                correctCount++;
            }
            LossValue += loss(y1[test_y[i]], 1);
              //先生私はコピペしました。なのでこの単位は必要ありません。


        }

        printf("%2d エポック　LOSS %2.4f ", l+1, LossValue / test_count);
        printf("%2.4f%%\n", 100.0 * correctCount/ test_count );

    }

    //学習結果保存
    save("fc1.dat",50,784, A1, b1);
    save("fc2.dat",100,50, A2, b2);
    save("fc3.dat",10,100, A3, b3);

    //メモリ解放
      free(y1);free(y2);
    free(dA1);free(db1);free(dA2);free(db2);free(dA3);free(db3);
    free(AVEdA1);free(AVEdb1);free(AVEdA2);free(AVEdb2);free(AVEdA3);free(AVEdb3);
    free(A1);free(b1);free(A2);free(b2);free(A3);free(b3);


    printf("\n");
    return 0;
}


