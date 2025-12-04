#include <cblas.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

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

void identity(vector<vector<float>>& I, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i][j] = (i == j) ? 1.0f : 0.0f;
}

// 2D -> 1D
void flatten(const vector<vector<float>>& A, vector<float>& flat, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            flat[i * N + j] = A[i][j];
}

// 1D -> 2D
void unflatten(const vector<float>& flat, vector<vector<float>>& A, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = flat[i * N + j];
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

    // B = A^T / (||A||_1 * ||A||_inf)
    vector B(N, vector<float>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j][i] = A[i][j];

    float scalar = 1.0f / (norm1(A, N) * normInf(A, N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i][j] *= scalar;

    // Одномерные массивы для Бласа
    vector<float> A_flat(N * N), B_flat(N * N);
    flatten(A, A_flat, N);
    flatten(B, B_flat, N);

    // R = I - BA
    vector I(N, vector<float>(N));
    identity(I, N);

    vector<float> I_flat(N * N), BA_flat(N * N), R_flat(N * N);
    flatten(I, I_flat, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, B_flat.data(), N, A_flat.data(), N, 0.0f, BA_flat.data(), N);

    for (int i = 0; i < N * N; i++)
        R_flat[i] = I_flat[i] - BA_flat[i];

    // Sum = I + R + R^2 + ...
    vector<float> Sum_flat = I_flat; // Sum
    vector<float> buffer1 = I_flat;  // Rn
    vector<float> buffer2(N * N);    // temp

    for (int m = 1; m <= M; m++)
    {
        // buffer2 = buffer1 * R
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, buffer1.data(), N, R_flat.data(), N,
                    0.0f, buffer2.data(), N);

        // Sum += buffer2
        cblas_saxpy(N * N, 1.0f, buffer2.data(), 1, Sum_flat.data(), 1);

        swap(buffer1, buffer2);
    }

    // A^(-1) = Sum * B
    vector<float> Ainv_flat(N * N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, Sum_flat.data(), N, B_flat.data(), N, 0.0f, Ainv_flat.data(), N);

    vector Ainv(N, vector<float>(N));
    unflatten(Ainv_flat, Ainv, N);

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
