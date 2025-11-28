#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

// Транспонирование
void transpose(const vector<vector<float>>& A, vector<vector<float>>& AT, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            AT[j][i] = A[i][j];
}

// Норма ||A||_1
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

// Норма ||A||_inf
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

// Умножение матриц C = A * B (оптимизированный порядок i-k-j)
void matmul(const vector<vector<float>>& A, const vector<vector<float>>& B,
            vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0;

    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// Умножение матрицы на скаляр
void matscal(const vector<vector<float>>& A, float s,
             vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] * s;
}

// Сложение матриц C = A + B
void matadd(const vector<vector<float>>& A, const vector<vector<float>>& B,
            vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] + B[i][j];
}

// Единичная матрица
void Identity(vector<vector<float>>& I, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i][j] = (i == j) ? 1.0f : 0.0f;
}

// Вычитание матриц C = A - B
void matsub(const vector<vector<float>>& A, const vector<vector<float>>& B,
            vector<vector<float>>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] - B[i][j];
}

int main()
{
    int N = 1024;
    int M = 10;

    // Инициализация матрицы A
    vector A(N, vector<float>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i == j) ? 2.0f : 0.1f;

    auto start = chrono::high_resolution_clock::now();

    // Вычисление B = A^T / (||A||_1 * ||A||_inf)
    vector AT(N, vector<float>(N));
    transpose(A, AT, N);
    float n1 = norm1(A, N);
    float ninf = normInf(A, N);
    float scalar = 1.0f / (n1 * ninf);

    vector B(N, vector<float>(N));
    matscal(AT, scalar, B, N);

    // R = I - BA
    vector I(N, vector<float>(N));
    vector BA(N, vector<float>(N));
    vector R(N, vector<float>(N));
    Identity(I, N);
    matmul(B, A, BA, N);
    matsub(I, BA, R, N);

    // Вычисление суммы ряда: Sum = I + R + R^2 + ...
    vector Sum(N, vector<float>(N));
    vector Rn(N, vector<float>(N));
    Identity(Sum, N); // Sum = I
    Identity(Rn, N); // R^0 = I

    for (int m = 1; m <= M; m++)
    {
        vector temp(N, vector<float>(N));
        matmul(Rn, R, temp, N); // R^n = R^(n-1) * R
        Rn = temp;
        matadd(Sum, Rn, Sum, N); // Sum += R^n
    }

    // A^(-1) = Sum * B
    vector Ainv(N, vector<float>(N));
    matmul(Sum, B, Ainv, N);

    auto end = chrono::high_resolution_clock::now();

    // Вывод результата
    cout << "Inverse elements:\n";
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << Ainv[i][j] << " ";
        cout << "\n";
    }

    cout << "time: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms\n";

    return 0;
}
