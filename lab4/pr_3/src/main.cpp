#include <cblas.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

float norm1(const float* A, int N)
{
    float maxSum = 0;
    for (int j = 0; j < N; j++)
    {
        float sum = cblas_sasum(N, &A[j], N);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

float normInf(const float* A, int N)
{
    float maxSum = 0;
    for (int i = 0; i < N; i++)
    {
        float sum = cblas_sasum(N, &A[i * N], 1);
        if (sum > maxSum) maxSum = sum;
    }
    return maxSum;
}

void identity(float* I, int N)
{
    for (int i = 0; i < N * N; i++) I[i] = 0;
    for (int i = 0; i < N; i++) I[i * N + i] = 1.0f;
}

int main()
{
    int N = 1024;
    int M = 10;

    vector<float> A(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = (i == j) ? 2.0f : 0.1f;

    auto start = chrono::high_resolution_clock::now();

    // B = A^T / (||A||_1 * ||A||_inf)
    vector<float> B(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];

    float scalar = 1.0f / (norm1(A.data(), N) * normInf(A.data(), N));
    cblas_sscal(N * N, scalar, B.data(), 1);

    // R = I - BA
    vector<float> I(N * N), BA(N * N), R(N * N);
    identity(I.data(), N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, B.data(), N, A.data(), N, 0.0f, BA.data(), N);

    for (int i = 0; i < N * N; i++)
        R[i] = I[i] - BA[i];

    // Sum = I + R + R^2 + ...
    vector<float> Sum(N * N), Rn(N * N), temp(N * N);
    identity(Sum.data(), N);
    identity(Rn.data(), N);

    for (int m = 1; m <= M; m++)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N, 1.0f, Rn.data(), N, R.data(), N, 0.0f, temp.data(), N); //temp=Rn×R
        Rn = temp; //Rn=R^m
        cblas_saxpy(N * N, 1.0f, Rn.data(), 1, Sum.data(), 1); // Sum=Sum+Rn
    }

    // A^(-1) = Sum * B
    vector<float> Ainv(N * N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, Sum.data(), N, B.data(), N, 0.0f, Ainv.data(), N);

    auto end = chrono::high_resolution_clock::now();

    cout << "Элементы обратной матрицы:\n";
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << Ainv[i * N + j] << " ";
        cout << "\n";
    }

    cout << "Время: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " мс\n";
    return 0;
}
