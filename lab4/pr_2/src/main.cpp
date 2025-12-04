#include <xmmintrin.h>  // SSE
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

// Транспонирование в двумерный массив
void transpose(const vector<vector<float>>& A, vector<vector<float>>& AT, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            AT[j][i] = A[i][j];
}

float norm1(const vector<vector<float>>& A, int N)
{
    float maxSum = 0;
    for (int j = 0; j < N; j++)
    {
        float sum = 0;
        for (int i = 0; i < N; i++)
            sum += fabs(A[i][j]);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

float normInf(const vector<vector<float>>& A, int N)
{
    float maxSum = 0;
    for (int i = 0; i < N; i++)
    {
        float sum = 0;
        for (int j = 0; j < N; j++)
            sum += fabs(A[i][j]);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

// Векторизованное умножение матриц
void matmul_sse(const vector<vector<float>>& A, const vector<vector<float>>& B,
                vector<vector<float>>& C, int N)
{
    // Обнуляем C
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0;

    // Транспонируем B для доступа к столбцам
    vector BT(N, vector<float>(N));
    transpose(B, BT, N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k = 0;

            for (; k <= N - 4; k += 4)
            {
                __m128 av = _mm_loadu_ps(&A[i][k]);      // A[i][k..k+3]
                __m128 bv = _mm_loadu_ps(&BT[j][k]);     // B^T[j][k..k+3] = B[k..k+3][j]
                sum = _mm_add_ps(sum, _mm_mul_ps(av, bv));
            }

            // Суммируем 4 элемента из sum
            float temp[4];
            _mm_storeu_ps(temp, sum);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Остаток
            for (; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

// Векторизованное умножение на скаляр
void matscal_sse(const vector<vector<float>>& A, float s, vector<vector<float>>& C, int N)
{
    __m128 sv = _mm_set1_ps(s);
    for (int i = 0; i < N; i++)
    {
        int j = 0;
        for (; j <= N - 4; j += 4)
        {
            __m128 av = _mm_loadu_ps(&A[i][j]);
            __m128 cv = _mm_mul_ps(av, sv);
            _mm_storeu_ps(&C[i][j], cv);
        }
        for (; j < N; j++)
            C[i][j] = A[i][j] * s;
    }
}

// Векторизованное сложение
void matadd_sse(const vector<vector<float>>& A, const vector<vector<float>>& B,
                vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
    {
        int j = 0;
        for (; j <= N - 4; j += 4)
        {
            __m128 av = _mm_loadu_ps(&A[i][j]);
            __m128 bv = _mm_loadu_ps(&B[i][j]);
            __m128 cv = _mm_add_ps(av, bv);
            _mm_storeu_ps(&C[i][j], cv);
        }
        for (; j < N; j++)
            C[i][j] = A[i][j] + B[i][j];
    }
}

void matsub_sse(const vector<vector<float>>& A, const vector<vector<float>>& B,
                vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
    {
        int j = 0;
        for (; j <= N - 4; j += 4)
        {
            __m128 av = _mm_loadu_ps(&A[i][j]);
            __m128 bv = _mm_loadu_ps(&B[i][j]);
            __m128 cv = _mm_sub_ps(av, bv);
            _mm_storeu_ps(&C[i][j], cv);
        }
        for (; j < N; j++)
            C[i][j] = A[i][j] - B[i][j];
    }
}

void Identity(vector<vector<float>>& I, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i][j] = (i == j) ? 1.0f : 0.0f;
}

int main()
{
    int N = 2048;
    int M = 10;

    vector A(N, vector<float>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i == j) ? 2.0f : 0.1f;

    auto start = chrono::high_resolution_clock::now();

    vector AT(N, vector<float>(N));
    vector B(N, vector<float>(N));
    vector BA(N, vector<float>(N));
    vector I(N, vector<float>(N));
    vector R(N, vector<float>(N));

    transpose(A, AT, N);
    float scalar = 1.0f / (norm1(A, N) * normInf(A, N));
    matscal_sse(AT, scalar, B, N);

    Identity(I, N);
    matmul_sse(B, A, BA, N);
    matsub_sse(I, BA, R, N);

    vector Sum(N, vector<float>(N));
    vector Rn(N, vector<float>(N));
    vector temp(N, vector<float>(N));

    Identity(Sum, N);
    Identity(Rn, N);

    for (int m = 1; m <= M; m++)
    {
        matmul_sse(Rn, R, temp, N);
        Rn = temp;
        matadd_sse(Sum, Rn, Sum, N);
    }

    vector Ainv(N, vector<float>(N));
    matmul_sse(Sum, B, Ainv, N);

    auto end = chrono::high_resolution_clock::now();

    cout << "Элементы обратной матрицы:\n";
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << Ainv[i][j] << " ";
        cout << "\n";
    }

    cout << "Время: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " мс\n";
    return 0;
}
