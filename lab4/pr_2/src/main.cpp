#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

void transpose(const float* A, float* AT, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            AT[j*N + i] = A[i*N + j];
}

float norm1(const float* A, int N) {
    float maxSum = 0;
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int i = 0; i < N; i++)
            sum += fabs(A[i*N + j]);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

float normInf(const float* A, int N) {
    float maxSum = 0;
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++)
            sum += fabs(A[i*N + j]);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

// Векторизованное умножение матриц
void matmul_sse(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i*N + j] = 0;

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m128 av = _mm_set1_ps(A[i*N + k]);
            int j = 0;
            for (; j <= N - 4; j += 4) {
                __m128 bv = _mm_loadu_ps(&B[k*N + j]);
                __m128 cv = _mm_loadu_ps(&C[i*N + j]);
                cv = _mm_add_ps(cv, _mm_mul_ps(av, bv));
                _mm_storeu_ps(&C[i*N + j], cv);
            }
            for (; j < N; j++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
    }
}

// Векторизованное умножение на скаляр
void matscal_sse(const float* A, float s, float* C, int N) {
    __m128 sv = _mm_set1_ps(s);
    int i = 0;
    int total = N * N;
    for (; i <= total - 4; i += 4) {
        __m128 av = _mm_loadu_ps(&A[i]);
        __m128 cv = _mm_mul_ps(av, sv);
        _mm_storeu_ps(&C[i], cv);
    }
    for (; i < total; i++)
        C[i] = A[i] * s;
}

// Векторизованное сложение
void matadd_sse(const float* A, const float* B, float* C, int N) {
    int total = N * N;
    int i = 0;
    for (; i <= total - 4; i += 4) {
        __m128 av = _mm_loadu_ps(&A[i]);
        __m128 bv = _mm_loadu_ps(&B[i]);
        __m128 cv = _mm_add_ps(av, bv);
        _mm_storeu_ps(&C[i], cv);
    }
    for (; i < total; i++)
        C[i] = A[i] + B[i];
}

void matsub_sse(const float* A, const float* B, float* C, int N) {
    int total = N * N;
    int i = 0;
    for (; i <= total - 4; i += 4) {
        __m128 av = _mm_loadu_ps(&A[i]);
        __m128 bv = _mm_loadu_ps(&B[i]);
        __m128 cv = _mm_sub_ps(av, bv);
        _mm_storeu_ps(&C[i], cv);
    }
    for (; i < total; i++)
        C[i] = A[i] - B[i];
}

void Identity(float* I, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i*N + j] = (i == j) ? 1.0f : 0.0f;
}

int main() {
    int N = 1024;
    int M = 10;

    vector<float> A(N*N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i*N + j] = (i == j) ? 2.0f : 0.1f;

    auto start = chrono::high_resolution_clock::now();

    vector<float> AT(N*N), B(N*N), BA(N*N), I(N*N), R(N*N);
    transpose(A.data(), AT.data(), N);
    float scalar = 1.0f / (norm1(A.data(), N) * normInf(A.data(), N));
    matscal_sse(AT.data(), scalar, B.data(), N);

    Identity(I.data(), N);
    matmul_sse(B.data(), A.data(), BA.data(), N);
    matsub_sse(I.data(), BA.data(), R.data(), N);

    vector<float> Sum(N*N), Rn(N*N), temp(N*N);
    Identity(Sum.data(), N);
    Identity(Rn.data(), N);

    for (int m = 1; m <= M; m++) {
        matmul_sse(Rn.data(), R.data(), temp.data(), N);
        Rn = temp;
        matadd_sse(Sum.data(), Rn.data(), Sum.data(), N);
    }

    vector<float> Ainv(N*N);
    matmul_sse(Sum.data(), B.data(), Ainv.data(), N);

    auto end = chrono::high_resolution_clock::now();

    cout << "Элементы обратной матрицы:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            cout << Ainv[i*N + j] << " ";
        cout << "\n";
    }

    cout << "Время: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " мс\n";
    return 0;
}
