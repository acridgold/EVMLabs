#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

static uint64_t rdtsc_precise() {
    unsigned int lo, hi;
    unsigned aux;
    __asm__ __volatile__(
        "rdtscp\n\t"
        : "=a"(lo), "=d"(hi), "=c"(aux)
        :
        : ); // Отсекаем слева
    __asm__ __volatile__("cpuid" ::: "%rax","%rbx","%rcx","%rdx"); // Строго отсекаем справа
    return ((uint64_t)hi << 32) | lo;
}

void make_forward(int *a, int N) {
    for (int i = 0; i < N - 1; ++i)
        a[i] = i + 1;
    a[N - 1] = 0;
}

void make_backward(int *a, int N) {
    a[0] = N - 1;
    for (int i = N - 1; i > 0; --i)
        a[i] = i - 1;
}

// случайный цикл (Фишер–Йетс)
void make_random_cycle(int *a, int N) {
    for (int i = 0; i < N; ++i)
        a[i] = i;

    for (int i = N - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        swap(a[i], a[j]);
    }
}

double measure_ticks_per_access_once(int *x, int N, long long K) {
    // прогрев
    volatile int k = 0;
    for (long long i = 0; i < N * K; ++i)
        k = x[k];
    if (k == 12345) printf("warmup\n");

    uint64_t t1 = rdtsc_precise();
    k = 0;
    for (long long i = 0; i < N * K; ++i)
        k = x[k];
    uint64_t t2 = rdtsc_precise();

    if (k == 12345) printf("use\n");

    return double(t2 - t1) / double(N * K);
}

// медиана
double measure_ticks_per_access(int *x, int N, long long K, int repeats = 7) {
    vector<double> vals;
    vals.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
        vals.push_back(measure_ticks_per_access_once(x, N, K));

    sort(vals.begin(), vals.end());
    return vals[repeats / 2];
}

double measure_random_ticks_per_access(int *buf, int N, long long K, int repeats = 7) {
    vector<double> vals;
    vals.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        make_random_cycle(buf, N);
        vals.push_back(measure_ticks_per_access_once(buf, N, K));
    }
    sort(vals.begin(), vals.end());
    return vals[repeats / 2];
}

int main() {
    srand((unsigned)time(nullptr));

#ifdef _WIN32
    // фиксируем процесс на одном ядре и поднимаем приоритет
    HANDLE hProc = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();

    DWORD_PTR mask = 1ull << 2;
    SetProcessAffinityMask(hProc, mask);
    SetThreadAffinityMask(hProc, mask);
    SetThreadAffinityMask(hThread, mask);

    SetPriorityClass(hProc, REALTIME_PRIORITY_CLASS);
    SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
#endif

    // разгон CPU ~1 сек
    {
        const int M = 512;
        double s = 0.0;
        clock_t t0 = clock();
        while (double(clock() - t0) / CLOCKS_PER_SEC < 1.0) {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < M; ++j)
                    s += i * j;
        }
        if ((int)s == 42) printf("turbo\n");
    }

    printf("#N\tforward\treverse\trandom (ticks/access)\n");

    int N = 256;

    // идём до ~256 МБ
    const int Nmax = 64 * 1024 * 1024;

    // диапазон обращений
    const long long min_accesses = 20'000'000LL;
    const long long max_accesses = 500'000'000LL;

    while (N <= Nmax) {
        int *a = (int*)malloc((size_t)N * sizeof(int));
        if (!a) {
            printf("malloc failed at N=%d\n", N);
            break;
        }

        // подбираем K так, чтобы N*K было в [min_accesses, max_accesses]
        long long K = min_accesses / N;
        if (K < 1) K = 1;
        if (N*K > max_accesses) K = max_accesses / N;
        if (K < 1) K = 1;

        // прямой
        make_forward(a, N);
        double fwd = measure_ticks_per_access(a, N, K);

        // обратный
        make_backward(a, N);
        double bwd = measure_ticks_per_access(a, N, K);

        // случайный
        double rnd = measure_random_ticks_per_access(a, N, K);

        printf("%d\t%.3f\t%.3f\t%.3f\n", N, fwd, bwd, rnd);

        free(a);

        N = (int)floor(N * 1.2);
        if (N <= 0) N = 1;
    }

    return 0;
}
