#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <climits>
#include <exception>

using namespace std;
namespace fs = std::filesystem;

// =========================== PARAMS GA ========================================
struct GAParams {
    int POP_SIZE = 160;
    int GENERATIONS = 1200;
    int ELITE_SIZE = 14;
    double P_CROSS = 0.90;
    double P_MUT_START = 0.18;
    double P_MUT_END   = 0.06;
    int CAND_SIZE = 60;
    int LOCAL_SEARCH_FREQ = 40;
    int LOCAL_SEARCH_TOPK = 3;
    double OFFSPRING_TS_PROB = 0.06;
    int DIVERSITY_CHECK_FREQ = 70;
    double DIVERSITY_THRESHOLD = 0.12;
    int STAGNATION_LIMIT = 240;
    int KICK_START = 260;
};
GAParams PARAMS;

// =========================== DATA STRUCTURES ==================================
int N = 0;
vector<int> A, B;

// Helper pentru acces rapid la matrice
static inline int Aat(int i, int j) { return A[i * N + j]; }
static inline int Bat(int i, int j) { return B[i * N + j]; }

vector<vector<int>> candidates;

struct Individual {
    vector<int> chromosome;
    long long cost = LLONG_MAX; // Initializam cu MAX pentru a nu fi considerat "bun" daca e necalculat
    bool operator<(const Individual& other) const { return cost < other.cost; }
};

static mt19937 rng((uint32_t)chrono::high_resolution_clock::now().time_since_epoch().count());
static long long REPAIR_COUNT = 0;

// Random helpers
static inline int rand_int(int lo, int hi) {
    uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}
static inline double rand01() {
    return uniform_real_distribution<double>(0.0, 1.0)(rng);
}
static inline int clampi(int x, int lo, int hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}
static inline double clampd(double x, double lo, double hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}
static inline int lerp_int(double t, int a, int b) {
    return (int)llround(a + t * (double)(b - a));
}
static inline double lerp_double(double t, double a, double b) {
    return a + t * (b - a);
}

// =========================== TUNING GA ========================================
static void tune_params_for_N(int N) {
    double t = clampd((N - 30.0) / 270.0, 0.0, 1.0);
    PARAMS.POP_SIZE = clampi(lerp_int(t, 190, 120), 100, 220);
    PARAMS.GENERATIONS = clampi(lerp_int(t, 1700, 950), 700, 2000);
    PARAMS.ELITE_SIZE = clampi(PARAMS.POP_SIZE / 10, 8, 20);
    PARAMS.CAND_SIZE = min(clampi((int)llround(0.30 * (double)N), 25, 90), N - 1);
    PARAMS.LOCAL_SEARCH_FREQ = clampi(lerp_int(t, 35, 110), 25, 140);
    PARAMS.LOCAL_SEARCH_TOPK = clampi(lerp_int(t, 5, 3), 2, 6);
    PARAMS.OFFSPRING_TS_PROB = clampd(lerp_double(t, 0.12, 0.04), 0.02, 0.18);
    PARAMS.DIVERSITY_CHECK_FREQ = clampi(lerp_int(t, 60, 95), 40, 120);
    PARAMS.DIVERSITY_THRESHOLD = clampd(lerp_double(t, 0.13, 0.10), 0.08, 0.18);
    PARAMS.STAGNATION_LIMIT = clampi(lerp_int(t, 140, 260), 100, 320);
    PARAMS.KICK_START = clampi(lerp_int(t, 90, 300), 60, 360);
    PARAMS.P_MUT_START = clampd(lerp_double(t, 0.18, 0.14), 0.08, 0.25);
    PARAMS.P_MUT_END   = clampd(lerp_double(t, 0.06, 0.045), 0.02, 0.10);
}

// =========================== LOAD & COST ======================================
bool loadInstance(const fs::path& filepath) {
    ifstream infile(filepath);
    if (!infile.is_open()) return false;
    if (!(infile >> N)) return false;

    A.assign(N * N, 0);
    B.assign(N * N, 0);

    for (int i = 0; i < N * N; i++) infile >> A[i];
    for (int i = 0; i < N * N; i++) infile >> B[i];
    return true;
}

long long calculate_cost(const vector<int>& p) {
    long long total = 0;
    for (int i = 0; i < N; i++) {
        int pi = p[i];
        const int baseA = i * N;
        const int baseBpi = pi * N;
        for (int j = 0; j < N; j++) {
            total += 1LL * A[baseA + j] * B[baseBpi + p[j]];
        }
    }
    return total;
}

// =========================== DELTA ============================================
long long calculate_delta(int r, int s, const vector<int>& p) {
    long long delta = 0;
    int pr = p[r], ps = p[s];

    for (int k = 0; k < N; k++) {
        if (k == r || k == s) continue;
        int pk = p[k];
        delta -= 1LL * Aat(r, k) * Bat(pr, pk) + Aat(k, r) * Bat(pk, pr);
        delta += 1LL * Aat(r, k) * Bat(ps, pk) + Aat(k, r) * Bat(pk, ps);
        delta -= 1LL * Aat(s, k) * Bat(ps, pk) + Aat(k, s) * Bat(pk, ps);
        delta += 1LL * Aat(s, k) * Bat(pr, pk) + Aat(k, s) * Bat(pk, pr);
    }
    delta -= 1LL * Aat(r, s) * Bat(pr, ps) + Aat(s, r) * Bat(ps, pr);
    delta += 1LL * Aat(r, s) * Bat(ps, pr) + Aat(s, r) * Bat(pr, ps);
    delta -= 1LL * Aat(r, r) * Bat(pr, pr) + Aat(s, s) * Bat(ps, ps);
    delta += 1LL * Aat(r, r) * Bat(ps, ps) + Aat(s, s) * Bat(pr, pr);
    return delta;
}

static inline long long delta_3cycle(int i, int j, int k, const vector<int>& p) {
    int pi = p[i], pj = p[j], pk = p[k];
    long long delta = 0;

    for (int t = 0; t < N; t++) {
        if (t == i || t == j || t == k) continue;
        int pt = p[t];
        delta -= Aat(i,t)*Bat(pi,pt) + Aat(t,i)*Bat(pt,pi);
        delta += Aat(i,t)*Bat(pk,pt) + Aat(t,i)*Bat(pt,pk);
        delta -= Aat(j,t)*Bat(pj,pt) + Aat(t,j)*Bat(pt,pj);
        delta += Aat(j,t)*Bat(pi,pt) + Aat(t,j)*Bat(pt,pi);
        delta -= Aat(k,t)*Bat(pk,pt) + Aat(t,k)*Bat(pt,pk);
        delta += Aat(k,t)*Bat(pj,pt) + Aat(t,k)*Bat(pt,pj);
    }
    delta -= Aat(i,j)*Bat(pi,pj) + Aat(j,i)*Bat(pj,pi);
    delta -= Aat(i,k)*Bat(pi,pk) + Aat(k,i)*Bat(pk,pi);
    delta -= Aat(j,k)*Bat(pj,pk) + Aat(k,j)*Bat(pk,pj);
    delta += Aat(i,j)*Bat(pk,pi) + Aat(j,i)*Bat(pi,pk);
    delta += Aat(i,k)*Bat(pk,pj) + Aat(k,i)*Bat(pj,pk);
    delta += Aat(j,k)*Bat(pi,pj) + Aat(k,j)*Bat(pj,pi);

    // Diagonale
    delta -= 1LL * Aat(i, i) * Bat(pi, pi);
    delta -= 1LL * Aat(j, j) * Bat(pj, pj);
    delta -= 1LL * Aat(k, k) * Bat(pk, pk);
    delta += 1LL * Aat(i, i) * Bat(pk, pk);
    delta += 1LL * Aat(j, j) * Bat(pi, pi);
    delta += 1LL * Aat(k, k) * Bat(pj, pj);

    return delta;
}

// =========================== VALIDATION & REPAIR ==============================
bool is_valid_permutation(const vector<int>& p) {
    if ((int)p.size() != N) return false;
    vector<int> freq(N, 0);
    for (int x : p) {
        if (x < 0 || x >= N) return false;
        if (++freq[x] > 1) return false;
    }
    return true;
}

void repair_permutation(vector<int>& p) {
    vector<int> freq(N, 0);
    vector<int> missing;
    for (int x : p) if (0 <= x && x < N) freq[x]++;
    for (int v = 0; v < N; v++) if (freq[v] == 0) missing.push_back(v);
    int miss_idx = 0;
    for (int i = 0; i < N; i++) {
        if (p[i] < 0 || p[i] >= N || freq[p[i]] > 1) {
            if (0 <= p[i] && p[i] < N) freq[p[i]]--;
            p[i] = missing[miss_idx++];
        }
    }
}

// =========================== CANDIDATE LIST ===================================
void build_candidates(int C) {
    candidates.assign(N, {});
    C = min(C, N - 1);
    if (C <= 0) return;
    for (int f = 0; f < N; f++) {
        vector<pair<int,int>> v;
        for (int g = 0; g < N; g++) {
            if (g == f) continue;
            v.push_back({ -(Aat(f, g) + Aat(g, f)), g });
        }
        nth_element(v.begin(), v.begin() + C, v.end());
        v.resize(C);
        sort(v.begin(), v.end());
        for (auto &pr : v) candidates[f].push_back(pr.second);
    }
}

// =========================== GA OPERATORS =====================================
double calculate_diversity(const vector<Individual>& pop) {
    if (pop.size() < 2) return 1.0;
    double total = 0.0;
    int comps = 0;
    size_t M = min((size_t)30, pop.size());
    for (size_t i = 0; i < M; i++) {
        for (size_t j = i + 1; j < M; j++) {
            int diff = 0;
            for (int k = 0; k < N; k++) diff += (pop[i].chromosome[k] != pop[j].chromosome[k]);
            total += (double)diff / N;
            comps++;
        }
    }
    return comps ? total / comps : 0.0;
}

void increase_diversity(vector<Individual>& pop) {
    int keep = min(PARAMS.ELITE_SIZE, (int)pop.size());
    vector<int> base_perm(N);
    iota(base_perm.begin(), base_perm.end(), 0);
    for (int i = keep; i < (int)pop.size(); i++) {
        pop[i].chromosome = base_perm;
        shuffle(pop[i].chromosome.begin(), pop[i].chromosome.end(), rng);
        pop[i].cost = calculate_cost(pop[i].chromosome);
    }
}

void crossover_ox(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
    int cx1 = rand_int(0, N - 1), cx2 = rand_int(0, N - 1);
    if (cx1 > cx2) swap(cx1, cx2);
    c1.chromosome.assign(N, -1); c2.chromosome.assign(N, -1);
    for (int i = cx1; i <= cx2; i++) {
        c1.chromosome[i] = p1.chromosome[i];
        c2.chromosome[i] = p2.chromosome[i];
    }
    auto fill = [&](Individual& c, const Individual& p) {
        vector<char> used(N, 0);
        for (int i = cx1; i <= cx2; i++) used[c.chromosome[i]] = 1;
        int pos = (cx2 + 1) % N;
        for (int t = 0; t < N; t++) {
            int idx = (cx2 + 1 + t) % N, g = p.chromosome[idx];
            if (!used[g]) { c.chromosome[pos] = g; used[g] = 1; pos = (pos + 1) % N; }
        }
    };
    fill(c1, p2); fill(c2, p1);
    
    c1.cost = LLONG_MAX; c2.cost = LLONG_MAX;
}

void crossover_pmx(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
    int cx1 = rand_int(0, N - 1), cx2 = rand_int(0, N - 1);
    if (cx1 > cx2) swap(cx1, cx2);
    c1.chromosome.assign(N, -1); c2.chromosome.assign(N, -1);
    for (int i = cx1; i <= cx2; i++) {
        c1.chromosome[i] = p1.chromosome[i];
        c2.chromosome[i] = p2.chromosome[i];
    }
    vector<int> pos1(N), pos2(N);
    for (int i = 0; i < N; i++) { pos1[p1.chromosome[i]] = i; pos2[p2.chromosome[i]] = i; }

    auto fill = [&](Individual& c, const Individual& pA, const Individual& pB, const vector<int>& posB) {
        for (int i = cx1; i <= cx2; i++) {
            int gene = pB.chromosome[i];
            bool exists = false;
            for(int j=cx1; j<=cx2; j++) if(c.chromosome[j] == gene) { exists=true; break; }
            if(exists) continue;
            int pos = i;
            while(c.chromosome[pos] != -1) pos = posB[pA.chromosome[pos]];
            c.chromosome[pos] = gene;
        }
        for(int i=0; i<N; i++) if(c.chromosome[i] == -1) c.chromosome[i] = pB.chromosome[i];
        c.cost = LLONG_MAX; // Cost invalid
    };
    fill(c1, p1, p2, pos2); fill(c2, p2, p1, pos1);
}

void mutate_swap(Individual& ind, double p_mut) {
    if (rand01() < p_mut) {
        int i = rand_int(0, N - 1), j = rand_int(0, N - 1);
        if (i != j) {
            // Recalculam costul full daca e invalid, inainte sa aplicam delta
            if (ind.cost == LLONG_MAX || ind.cost == -1) ind.cost = calculate_cost(ind.chromosome);

            ind.cost += calculate_delta(min(i,j), max(i,j), ind.chromosome);
            swap(ind.chromosome[i], ind.chromosome[j]);
        }
    }
}

// =========================== TABU SEARCH (CORRECTED) ==========================
void tabu_search_candidates(Individual& ind, int iters, int tenure_min, int tenure_max,
                            int sample_i, int random_tries, long long& best_seen_global,
                            double p_3swap = 0.0, int tries_3swap = 0)
{
    if (ind.cost == LLONG_MAX || ind.cost == -1) ind.cost = calculate_cost(ind.chromosome);
    long long cur = ind.cost;
    if (candidates.empty()) return;
    vector<int> tabu(N * N, 0);
    vector<int> best_perm = ind.chromosome;
    long long best_cost = cur;
    vector<int> pos(N);
    for (int i = 0; i < N; i++) pos[ind.chromosome[i]] = i;

    for (int it = 1; it <= iters; it++) {
        long long best_d = LLONG_MAX;
        int bi = -1, bj = -1, bk = -1;
        bool use3 = false;

        // 2-opt Candidates
        for (int si = 0; si < sample_i; si++) {
            int i = rand_int(0, N - 1);
            int fa = ind.chromosome[i];
            for (int g : candidates[fa]) {
                int j = pos[g];
                if (j == i) continue;
                int a = min(i, j), b = max(i, j);
                long long d = calculate_delta(a, b, ind.chromosome);
                if (tabu[ind.chromosome[a]*N + ind.chromosome[b]] > it && cur + d >= best_seen_global) continue;
                if (d < best_d) { best_d = d; bi = a; bj = b; use3 = false; }
            }
        }
        // 2-opt Random
        for (int t = 0; t < random_tries; t++) {
            int i = rand_int(0, N - 1), j = rand_int(0, N - 1);
            if (i == j) continue;
            int a = min(i, j), b = max(i, j);
            long long d = calculate_delta(a, b, ind.chromosome);
            if (tabu[ind.chromosome[a]*N + ind.chromosome[b]] > it && cur + d >= best_seen_global && cur + d >= best_cost) continue;
            if (d < best_d) { best_d = d; bi = a; bj = b; use3 = false; }
        }
        // 3-opt
        if (rand01() < p_3swap) {
            for (int t = 0; t < tries_3swap; t++) {
                int i = rand_int(0, N - 1), j = rand_int(0, N - 1), k = rand_int(0, N - 1);
                if (i == j || i == k || j == k) continue;
                long long d3 = delta_3cycle(i, j, k, ind.chromosome);
                if (d3 < best_d) { best_d = d3; bi = i; bj = j; bk = k; use3 = true; }
            }
        }

        // Apply
        if (bi == -1) { // Random move to escape
             int a = rand_int(0, N-1), b = rand_int(0, N-1);
             if(a==b) b=(a+1)%N;
             if(a>b) swap(a,b);
             bi=a; bj=b; best_d = calculate_delta(a, b, ind.chromosome); use3=false;
        }

        if (use3) {
            int val_i = ind.chromosome[bi], val_j = ind.chromosome[bj], val_k = ind.chromosome[bk];
            ind.chromosome[bi] = val_k; ind.chromosome[bj] = val_i; ind.chromosome[bk] = val_j;
            pos[val_k] = bi; pos[val_i] = bj; pos[val_j] = bk;
            int ten = rand_int(tenure_min, tenure_max);
            tabu[val_i*N+val_j] = it+ten; tabu[val_j*N+val_k] = it+ten; tabu[val_k*N+val_i] = it+ten;
        } else {
            int f1 = ind.chromosome[bi], f2 = ind.chromosome[bj];
            swap(ind.chromosome[bi], ind.chromosome[bj]);
            pos[f1] = bj; pos[f2] = bi;
            int ten = rand_int(tenure_min, tenure_max);
            tabu[f1*N+f2] = it+ten; tabu[f2*N+f1] = it+ten;
        }
        cur += best_d;
        ind.cost = cur;
        if (cur < best_cost) { best_cost = cur; best_perm = ind.chromosome; }
        if (cur < best_seen_global) best_seen_global = cur;
    }
    ind.chromosome = best_perm;
    ind.cost = best_cost;
}

// =========================== ALGORITMUL GENETIC ===============================
Individual runMemeticGA() {
    tune_params_for_N(N);
    build_candidates(PARAMS.CAND_SIZE);
    vector<Individual> pop(PARAMS.POP_SIZE);
    vector<int> base(N); iota(base.begin(), base.end(), 0);

    for (int i = 0; i < PARAMS.POP_SIZE; i++) {
        pop[i].chromosome = base;
        shuffle(pop[i].chromosome.begin(), pop[i].chromosome.end(), rng);
        pop[i].cost = calculate_cost(pop[i].chromosome);
    }
    sort(pop.begin(), pop.end());
    long long best_seen_global = pop[0].cost;
    Individual best_global = pop[0];

    // Initial TS
    for (int i = 0; i < min(8, PARAMS.POP_SIZE); i++)
        tabu_search_candidates(pop[i], clampi(2*N, 200, 900), clampi(N/14, 8, 22), clampi(N/7, 16, 45), clampi(N/10, 18, 40), clampi(N/18, 8, 22), best_seen_global);
    sort(pop.begin(), pop.end());
    best_global = pop[0];

    int no_improve = 0;

    for (int gen = 1; gen <= PARAMS.GENERATIONS; gen++) {
        if (gen % 50 == 0) cerr << "\r[GA] g=" << gen << " best=" << best_global.cost << " ni=" << no_improve << "   ";

        // Diversity
        if (gen % PARAMS.DIVERSITY_CHECK_FREQ == 0 && no_improve > PARAMS.STAGNATION_LIMIT) {
             if (calculate_diversity(pop) < PARAMS.DIVERSITY_THRESHOLD) { increase_diversity(pop); sort(pop.begin(), pop.end()); }
        }

        // Local Search periodic
        if (gen % PARAMS.LOCAL_SEARCH_FREQ == 0) {
            int topk = min(PARAMS.LOCAL_SEARCH_TOPK, (int)pop.size());
            int iters = 150 + (gen * 250) / PARAMS.GENERATIONS;
            if(no_improve > 80) iters += 150;
            for(int i=0; i<topk; i++) tabu_search_candidates(pop[i], iters, 10, 22, 24, 12, best_seen_global);
            sort(pop.begin(), pop.end());
        }

        if (pop[0].cost < best_global.cost) { best_global = pop[0]; no_improve = 0; } else no_improve++;

        // Kick
        if (no_improve >= 50) {
            int kick_swaps = 5 + (no_improve/50)*2;
            for(int idx=PARAMS.ELITE_SIZE; idx<(int)pop.size(); idx++) {
                for(int t=0; t<kick_swaps; t++) {
                     int a = rand_int(0, N-1), b = rand_int(0, N-1);
                     if(a!=b) swap(pop[idx].chromosome[a], pop[idx].chromosome[b]);
                }
                pop[idx].cost = calculate_cost(pop[idx].chromosome);
            }
            sort(pop.begin(), pop.end());
            if (pop[0].cost < best_global.cost) best_global = pop[0];
        }

        // New Pop
        vector<Individual> new_pop;
        for(int i=0; i<PARAMS.ELITE_SIZE; i++) new_pop.push_back(pop[i]);

        while((int)new_pop.size() < PARAMS.POP_SIZE) {
            int i1 = rand_int(0, PARAMS.POP_SIZE/2), i2 = rand_int(0, PARAMS.POP_SIZE/2);
            Individual c1, c2;
            if(rand01() < PARAMS.P_CROSS) {
                if(gen & 1) crossover_ox(pop[i1], pop[i2], c1, c2); else crossover_pmx(pop[i1], pop[i2], c1, c2);
                c1.cost = LLONG_MAX; c2.cost = LLONG_MAX; // Marcam ca invalid
            } else { c1 = pop[i1]; c2 = pop[i2]; }

            double p_mut = PARAMS.P_MUT_START + (PARAMS.P_MUT_END - PARAMS.P_MUT_START) * ((double)gen/PARAMS.GENERATIONS);
            if(no_improve > 100) p_mut *= 1.5;

            mutate_swap(c1, p_mut); mutate_swap(c2, p_mut);

            if(!is_valid_permutation(c1.chromosome)) {
                repair_permutation(c1.chromosome); c1.cost=LLONG_MAX;
            }
            if(!is_valid_permutation(c2.chromosome)) {
                repair_permutation(c2.chromosome); c2.cost=LLONG_MAX;
            }


            if (c1.cost == LLONG_MAX || c1.cost == -1) c1.cost = calculate_cost(c1.chromosome);
            if (c2.cost == LLONG_MAX || c2.cost == -1) c2.cost = calculate_cost(c2.chromosome);

            new_pop.push_back(c1); if((int)new_pop.size() < PARAMS.POP_SIZE) new_pop.push_back(c2);
        }
        pop = new_pop;
        sort(pop.begin(), pop.end());
    }

    // Final Intensification
    tabu_search_candidates(best_global, clampi(40*N, 2500, 10000), 12, 30, 30, 15, best_seen_global);
    return best_global;
}

// =========================== SIMULATED ANNEALING (QAP ADAPTED) ================
Individual runSimulatedAnnealing() {
    // Configurare SA
    const double MAX_SECONDS = 3.0;
    const int MAX_ITERS = 10000000;
    double T = 5000.0;
    double T_min = 0.001;
    double alpha = 0.9995;

    auto start_time = chrono::high_resolution_clock::now();

    // 1. Solutie initiala random
    Individual current;
    current.chromosome.resize(N);
    iota(current.chromosome.begin(), current.chromosome.end(), 0);
    shuffle(current.chromosome.begin(), current.chromosome.end(), rng);
    current.cost = calculate_cost(current.chromosome);

    Individual best = current;
    int iters = 0;

    // 2. Bucla SA
    while (T > T_min && iters < MAX_ITERS) {

        // Verificam timpul la fiecare 5000 iteratii
        if ((iters & 4095) == 0) {
            auto now = chrono::high_resolution_clock::now();
            if (chrono::duration<double>(now - start_time).count() > MAX_SECONDS) break;
        }

        // Mutatie: Swap 2 indecsi
        int i = rand_int(0, N - 1);
        int j = rand_int(0, N - 1);
        while (i == j) j = rand_int(0, N - 1); // asigura i != j

        // Calcul delta eficient
        long long delta = calculate_delta(min(i, j), max(i, j), current.chromosome);

        // Acceptare
        bool accept = false;
        if (delta < 0) {
            accept = true;
        } else {
            // Boltzmann
            if (rand01() < exp(-delta / T)) {
                accept = true;
            }
        }

        if (accept) {
            // Aplicam swap
            swap(current.chromosome[i], current.chromosome[j]);
            current.cost += delta;

            // Actualizam best global
            if (current.cost < best.cost) {
                best = current;
            }
        }

        T *= alpha;
        iters++;
    }

    return best;
}


// =========================== UTIL =============================================
static bool open_for_write(ofstream& out, const fs::path& p) {
    out.open(p, ios::out);
    if (!out) { cerr << "EROARE scriere: " << p.string() << "\n"; return false; }
    return true;
}

void write_stats(const fs::path& folder, const string& fileName,
                 const vector<pair<long long, double>>& results,
                 const vector<Individual>& solutions, int nr_runs)
{
    vector<long long> costs;
    double sum_time = 0.0;
    long long best_c = LLONG_MAX, worst_c = LLONG_MIN;
    int best_idx = -1;

    for (int i=0; i<results.size(); i++) {
        long long c = results[i].first;
        costs.push_back(c);
        sum_time += results[i].second;
        if(c < best_c) { best_c = c; best_idx = i; }
        if(c > worst_c) worst_c = c;
    }
    sort(costs.begin(), costs.end());
    long long median = costs[costs.size()/2];
    double avg = accumulate(costs.begin(), costs.end(), 0.0) / costs.size();

    ofstream sf(folder / "statistics.txt");
    sf << "Instance: " << fileName << "\nRuns: " << nr_runs << "\n";
    sf << "Best:   " << best_c << "\n";
    sf << "Avg:    " << avg << "\n";
    sf << "Median: " << median << "\n";
    sf << "Worst:  " << worst_c << "\n";
    sf << "TimeAvg:" << sum_time/nr_runs << "s\n";
    if(best_idx!=-1) {
        sf << "\nBest Permutation:\n";
        for(int x : solutions[best_idx].chromosome) sf << x+1 << " ";
    }

    ofstream af(folder / "all_runs.txt");
    for(size_t i=0; i<results.size(); i++)
        af << "Run " << i+1 << ": " << results[i].first << " (" << results[i].second << "s)\n";
}

// =========================== MAIN =============================================
int main() {
    const int NR_RUNS = 30;
    fs::path inputFolder = "InputFolder";
    fs::path outputRoot  = "OutputFolder";
    fs::create_directories(outputRoot);

    vector<fs::path> files;
    try {
        for (const auto& entry : fs::directory_iterator(inputFolder))
            if (entry.is_regular_file() && entry.path().extension() == ".dat") files.push_back(entry.path());
    } catch (...) {}
    if (files.empty()) { cerr << "Nu exista fisiere .dat in InputFolder!\n"; return 1; }
    sort(files.begin(), files.end());

    for (const auto& filePath : files) {
        string fileName = filePath.filename().string();
        if (!loadInstance(filePath)) continue;

        cerr << "\n================ " << fileName << " (N=" << N << ") ================\n";
        fs::path instDir = outputRoot / fileName;
        fs::path dirGA = instDir / "GA";
        fs::path dirSA = instDir / "SA";
        fs::create_directories(dirGA);
        fs::create_directories(dirSA);

        cerr << "-> Rulare GA (" << NR_RUNS << " ori)...\n";
        vector<pair<long long, double>> resGA;
        vector<Individual> solGA;
        for(int r=0; r<NR_RUNS; r++) {
            auto t1 = chrono::high_resolution_clock::now();
            Individual sol = runMemeticGA();
            auto t2 = chrono::high_resolution_clock::now();
            double d = chrono::duration<double>(t2-t1).count();
            resGA.push_back({sol.cost, d});
            solGA.push_back(sol);
            cerr << "\r   GA Run " << r+1 << "/" << NR_RUNS << ": " << sol.cost << " (" << d << "s)    ";
        }
        write_stats(dirGA, fileName, resGA, solGA, NR_RUNS);
        cerr << "\n";

        cerr << "-> Rulare SA (" << NR_RUNS << " ori)...\n";
        vector<pair<long long, double>> resSA;
        vector<Individual> solSA;
        for(int r=0; r<NR_RUNS; r++) {
            auto t1 = chrono::high_resolution_clock::now();
            Individual sol = runSimulatedAnnealing();
            auto t2 = chrono::high_resolution_clock::now();
            double d = chrono::duration<double>(t2-t1).count();
            resSA.push_back({sol.cost, d});
            solSA.push_back(sol);
            cerr << "\r   SA Run " << r+1 << "/" << NR_RUNS << ": " << sol.cost << " (" << d << "s)    ";
        }
        write_stats(dirSA, fileName, resSA, solSA, NR_RUNS);
        cerr << "\n";
    }

    return 0;
}