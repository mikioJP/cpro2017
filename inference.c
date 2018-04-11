//
// Created by 小林　幹旺 on 2017/06/28.
//

#include "nn.h"
#include <time.h>
#define rnd() rand() / (RAND_MAX + 1.0) // [0, 1) の一様乱数を返す

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

//学習係数を読み込み
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

//交差エントロピー
float loss(const float y, unsigned char t){
    return (float) (-1 * t * log(y + 0.0000001));
}

// argv[1][2][3]には学習結果を保存したファイルをfc1から順に入力、argv[4]には推論のために使うファイル名を指定、そこに推論に使う
//bmpファイルを保存、読み込みを行う。
int main(int argc, char * argv[]){
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

    //メモリ確保
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    float *y = malloc(sizeof(float)*10);
    int i;

    srand((unsigned)time(NULL));
    i = rnd() * train_count;                      // 乱数によって選んだ訓練データを1 つargv[4]
    save_mnist_bmp(train_x + 784*i, argv[4], i);  // で指定した名前で保存

    float *x = load_mnist_bmp(argv[4]);           // 保存したBMP 画像を読み込む
    load(argv[1], 50, 784, A1, b1);               // 各FC 層のパラメータをファイルから読み込む
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    // 読み込んだパラメータを用いて推論を行い、
    printf("result = %d\n answer = %d\n",       // 画像に対する推論結果および正解を表示する
           inference6(A1, b1, A2, b2, A3, b3, x, y), *(train_y + i));

    free(A1); free(b1); free(A2); free(b2); free(A3); free(b3); free(y);
    return 0;
}
