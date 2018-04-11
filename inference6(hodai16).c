//
// Created by 小林　幹旺 on 2017/06/26.
//

#include "nn.h"
#include <time.h>

void print(int n, int m, const float *x){
    for (int j = 0; j < n ; ++j) {
        for (int i = 0; i < m ; ++i) {
            printf("%.4lf ", x[i + n * j]);
        }
        printf("\n");
    }

    printf("\n");
}

void copy(int n, const float *x, float *y){
    for (int i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

void fc (int m, int n, const float *x, const float *A, const float *b, float *y){
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            y[i] +=  A[n*i+j]*x[j];
        }
        y[i]+=b[i];
    }
}

void relu (int n, const float *x, float *y){
    for (int i = 0; i < n; ++i) {
        if (x[i] > 0){
            y[i] = x[i];
        }else{
            y[i]=0;
        }
    }
}

void softmax(int n, const float *x,float *y){
    float x_max=0;
    for (int i = 0; i < n; ++i) {
        if (x[i] > x_max){
            x_max = x[i];
        }
    }

    float a =0;
    for (int i = 0; i < n; ++i) {
        a += exp(x[i]-x_max);
    }

    for (int j = 0; j < n ; ++j) {

        y[j]= (float)exp(x[j] - x_max) / a;
    }

}

void softMaxWithLoss_bwd(int n, const float *y, unsigned char t, float * dx) {

    float correctAns[n];
    for (int j = 0; j < n; ++j) {
        if (j == t){
            correctAns[j] = 1;
        }else{
            correctAns[j] = 0;
        }
    }
    for (int i = 0; i < n; ++i) {
        dx[i] =  y[i] - correctAns[i];
    }
}

void relu_bwd(int n, const float *x, const float *dy, float *dx ){
    for (int i = 0; i < n; ++i) {
        if (x[i] > 0){
            dx[i] = dy[i];
        }else{
            dx[i] = 0;
        }
    }

}

void fc_bwd(int m, int n, const float *x, const float *dy, const float *A, float *dA, float *db, float *dx  ) {

    for (int k = 0; k < m; ++k) {
        for (int i = 0; i < n; ++i) {

            int p = i + n *k;

            dA[p] = dy[k] * x[i];
            db[k] = dy[k];
            dx[k] += A[p] * dy[i];
        }

    }
}

void init (int n, float x, float *o){
    for (int i = 0; i < n; ++i) {
        o[i] = x;
    }
}

void rand_init(int n, float *o){
    for (int i = 0; i < n; ++i) {
        o[i] = -1 + (float)( rand() * (1 + 1 + 1.0) / (1.0 + RAND_MAX) );
    }
}

void shuffle(int n, int *x){
    srand(time(NULL));
    for (int i = 0; i < n ; ++i) {
        int num =0 + (int)( rand() * (n - 0 + 1.0) / (1.0 + RAND_MAX) );
        int temp = x[num];
        x[num] = x[i];
        x[i] = temp;
    }
}


int inference6 (const float *A1,const float *b1,
                const float *A2,const float *b2,
                const float *A3,const float *b3,
                const float *x,float *y){

    float y1[50] = {0};
    float y2[100] = {0};
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2,A3, b3, y);
    softmax(10, y, y);

    float max = 0.0;
    int index = 0;

    for (int i = 0; i < 10 ; ++i) {
        if (max < y[i]){
            max = y[i];
            index = i;
        }
    }
    return  index;
}


void backward6(const float *A1, const float *b1,
               const float *A2, const float *b2,
               const float *A3, const float *b3,
               const float *x, unsigned char t,
               float *dA1,  float * db1,
               float *dA2,  float * db2,
               float *dA3,  float * db3){

    float fcTemp1[784] ={0};
    float ReLUTemp1[50]={0};
    float fcTemp2[50]={0};
    float ReLUTemp2[100]={0};
    float fcTemp3[100]={0};

    float dx_fc1[784]={0};
    float dx_relu1[50]={0};
    float dx_fc2[50]={0};
    float dx_relu2[100]={0};
    float dx_fc3[100]={0};
    float dx_softMaxWithLoss[10]={0};

    float y[10]={0};

    //順伝播
    copy(784, x, fcTemp1);
    fc(50, 784, x, A1, b1, ReLUTemp1);
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


int main(){
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
    // を使用することができる．
    float correctCount = 0.0;

    srand(time(NULL));


    float *y = malloc(sizeof(float) * 10);
    if (y == NULL){
        free(y);
        printf("Error y");
    }
    float *dA1 = malloc(sizeof(float) * 784 * 50);
    if (dA1 == NULL){
        free(dA1);
        printf("Error dA1");
    }
    float *db1 = malloc(sizeof(float) * 50);
    if (db1 == NULL){
        free(db1);
        printf("Error db1");
    }
    float *dA2 = malloc(sizeof(float) * 50 * 100);
    if (dA2 == NULL){
        free(dA2);
        printf("Error dA2");
    }
    float *db2 = malloc(sizeof(float) * 100);
    if (db2 == NULL){
        free(db2);
        printf("Error db2");
    }
    float *dA3 = malloc(sizeof(float) * 100 * 10);
    if (dA3 == NULL){
        free(dA3);
        printf("Error dA3");
    }
    float *db3 = malloc(sizeof(float) * 10);
    if (db3 == NULL){
        free(db3);
        printf("Error db3");
    }


    float AVEdA1[784*50];
    float AVEdb1[50];
    float AVEdA2[50*100];
    float AVEdb2[100];
    float AVEdA3[100*10];
    float AVEdb3[10];

    float A1[784*50];
    float b1[50];
    float A2[50*100];
    float b2[100];
    float A3[100*10];
    float b3[10];

    //係数を[-1:1]で初期化
    rand_init(784 * 50, A1);
    rand_init(50, b1);
    rand_init(50 * 100 , A2);
    rand_init(100, b2);
    rand_init(100 * 10, A3);
    rand_init(10, b3);

    init(784*50, 0, AVEdA1);
    init(50, 0, AVEdb1);
    init(50*100, 0, AVEdA2);
    init(100, 0, AVEdb2);
    init(100*10, 0, AVEdA3);
    init(10, 0, AVEdb3);

    int batchSize =100;
    int index[train_count];

    for (int i = 0; i < train_count; ++i) {
        index[i] = i;
    }

    shuffle(train_count, index);

    for (int k = 0; k  < 1; ++k) {
        for (int j = 0; j < 50; ++j) {
    backward6(A1, b1,
              A2,b2,
              A3, b3,
              train_x + width*height*index[j+k*batchSize], train_y[index[j + k*batchSize]],
              dA1, db1,
              dA2, db2,
              dA3, db3);

    //print(50, 784, dA1);

        }
    }




    for (int j = 0; j < test_count; ++j) {
        if (inference6(A1_784_50_100_10,b1_784_50_100_10,
                       A2_784_50_100_10,b2_784_50_100_10,
                       A3_784_50_100_10,b3_784_50_100_10,
                       test_x+784*j,y) == test_y[j]){
            correctCount++;
        }
    }

    printf("%lf%%\n ", 100.0000 * correctCount / test_count);
    free(y);

    return 0;
}