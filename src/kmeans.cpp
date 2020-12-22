
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <string.h>
#include "kmeans.h"

static int randd(int d)
{
    return (int)(1.0 * rand() / (RAND_MAX + 1.0) * d);
}

static double randf(double s)
{
    return (1.0 * rand() / (RAND_MAX + 0.1) * s);
}

static double dist(double *feature1, double *feature2, int f)
{
    double d = 0.0;
    for (int i = 0; i < f; i++)
    {
        d += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
    }
    return sqrt(d);
}

static int nearest(double *s, double *cents, int c, int f, double *dd)
{
    int rc = -1;
    double d = 1e12;
    for (int i = 0; i < c; i++)
    {
        double t = dist(s, cents + i * f, f);
        if (t < d)
        {
            d = t;
            rc = i;
        }
    }
    // s
    if (dd)
        *dd = (d * d - 2) / -2;
    return rc;
}

static int nearest_acc(double *s, double *cents, int c, int f, double *dd, double *centers_dists, int oldcent, double min_sim)
{
    int rc = -1;
    double d = 1e12;
    // 三角不等式加速
    if (oldcent != -1) {
        double d_xc = dist(s, cents + oldcent * f, f);
        if (2*d_xc <= centers_dists[oldcent]) {
            rc = oldcent;
            d = d_xc;
        }
    }
    // 计算距离
    if (rc == -1) {
        for (int i = 0; i < c; i++)
        {
            double t = dist(s, cents + i * f, f);
            if (t < d)
            {
                d = t;
                rc = i;
            }
        }
    }
    // 判断是否满足约束 
    double sim = (d*d - 2) / -2;
    if (dd)
        *dd = sim;
    if (sim < min_sim)
        rc = -1;
    return rc;
}


static int load_centers_before(double* centers, char *centers_path, int f, int old_centers_ct)
{
    FILE * ip = NULL;
    if (NULL == (ip = fopen(centers_path, "r"))){
        fprintf(stderr, "can not open old center file.\n");
        return -1;
    }
    char buffer[4096] = {0};
    int n = 0;
    while (NULL != fgets(buffer, 4096, ip)){
        n += 1;
    }
    if (n != old_centers_ct) {
        fprintf(stderr, "old center count err.\n");
        return -2;
    }
    rewind(ip);
    int i = 0;
    fscanf(ip, "%lf", centers);
    while (!feof(ip)){
        fscanf(ip, "%lf", centers + (++i));
    }
    fclose(ip);
    return 0;
}


static int binary_search(double *v, int n, double s){
    int l, h, m;
    if (n < 0) {return -1;}
    if (n == 0 || v[0] > s){ return 0;}
    if (v[n - 1] <= s) {return n - 1;}
    l = 0, h = n;
    while (h > l) {
        if (h == l + 1){
            return (v[l] > s) ? l : h;
        }
        m = (h + l) / 2;
        if (v[m] > s && v[m - 1] <= s){
            return m;
        }
        else if (v[m] <= s){
            l = m + 1;
        }
        else if (v[m - 1] > s){
            h = m - 1;
        }
    }
    return h;
}


static int init_cents(double *m, int n, int f, int k, double *cents, double *centsA, int *centsC, int *cids, char* old_centers_path, int old_centers_ct, int new_centers_keep, double min_sim)
{
    int sampled_i, b = randd(n);
    double t = 0.0;
    double *d = (double *)malloc(sizeof(double) * n);
    double *cd = (double*)malloc(sizeof(double) * n);
    memset(d, 0, sizeof(double) * n);
    // init 1st center or old centers according to k
    if (k != 0) {
        memmove(cents, m + b * f, sizeof(double) * f);
        old_centers_ct = 1;
    } else {
        load_centers_before(cents, old_centers_path, f, old_centers_ct);
        k = old_centers_ct + new_centers_keep;
    }

    for (int i = 0; i < n; i++) {
        d[i] = dist(m + i * f, cents, f);
        cd[i] = d[i];
        if (i > 0){
            cd[i] += cd[i - 1];
        }
    }

    // init left centers
    for (int c = old_centers_ct; c < k; c++)
    {
        t = randf(cd[n - 1]);
        sampled_i = binary_search(cd, n, t);
        memmove(cents + c * f, m + sampled_i * f, sizeof(double) * f);
        for (int i = 0; i < n; i++){
            t = dist(m + i * f,  cents + c * f, f);
            if (t < d[i]) {d[i] = t; cids[i] = c;}
            cd[i] = d[i];
            if (i > 0){ cd[i] += cd[i - 1]; }
        }
    }
    free(cd); cd = NULL;

    memset(d, 0, sizeof(double) * n);
    // calculate clsid for all points and center points
    for (int i = 0; i < n; i++)
    {
        int cent = nearest(m + i * f, cents, k, f, d + i);
        if (d[i] < min_sim) {
            cids[i] = -1;
        } else {
            cids[i] = cent;
            centsC[cent] += 1;
            for (int j = 0; j < f; j++)
            {
                centsA[cent * f + j] += m[i * f + j];
            }
        }
    }
    free(d); d = NULL;

    for (int i = 0; i < k; i++)
    {
        if (centsC[i] == 0) {
            for (int j = 0; j < f; j++)
                cents[i * f + j] = 0.0;
            continue;
        }

        double mod = 0.0;
        for (int j = 0; j < f; j++) {
            cents[i * f + j] = centsA[i * f + j] / centsC[i];
            mod += cents[i * f + j] * cents[i * f + j];
        }
        if (mod != 0.0) {
            mod = sqrt(mod);
            for (int j = 0; j < f; j++) {
                cents[i * f + j] = cents[i * f + j] / mod;
            }
        } else {
            // all zeros
            continue;
        }

    }

    return 0;
}

// save centers
int save_centers(int centers_ct, int dim, double *centers, int *centersC, char* centers_path_save){
    FILE * op = NULL;
    if (NULL == (op = fopen(centers_path_save, "w"))){
        fprintf(stderr, "save centers err.\n");
        return -1;
    }
    for (int i = 0; i < centers_ct; i++) {
        fprintf(op, "%d\t%d", i, centersC[i]);
        for (int j = 0; j < dim; j++) {
            fprintf(op, "\t%.5f", centers[i * dim + j]);
        }
        fprintf(op, "\n");
    }
    fclose(op);
    return 0;
}

void train_slave(int start, int stop, int k, int f, int *c, double* d, int* new_centers, double* m, double* cents, double* centers_min_dists, double min_sim){
    for (int i = start; i < stop; i++) {
        new_centers[i] = nearest_acc(m + i * f, cents, k, f, d + i, centers_min_dists, c[i], min_sim);
    }
}

int kmeans(double *m, int n, int f, int k, int *c, char* old_centers_path, int old_centers_ct, int new_centers_keep, char* centers_path_save, double* scores, int threads, double min_sim)
{
    int k_bak = k;
    if (k == 0) {
        k = old_centers_ct + new_centers_keep;
    }
    double *cents = (double *)malloc(sizeof(double) * k * f);
    double *centsA = (double *)malloc(sizeof(double) * k * f);
    double *d = (double *)malloc(sizeof(double) * n);
    memset(d, 0, sizeof(double) * n);

    int *centsC = (int *)malloc(sizeof(int) * k);

    // center dist matrix
    double *centers_min_dists = (double *)malloc(sizeof(double) * k);

    int *new_centers = (int *)malloc(sizeof(int) * n);

    memset(cents, 0, sizeof(double) * k * f);
    memset(centsA, 0, sizeof(double) * k * f);
    memset(centsC, 0, sizeof(int) * k);
    memset(new_centers, -1, sizeof(int) * n);

    init_cents(m, n, f, k_bak, cents, centsA, centsC, c, old_centers_path, old_centers_ct, new_centers_keep, min_sim);

    int niter = 0;
    while (niter < 100)
    {
        // calcuate center dist matrix
        for (int i = 0; i < k; i++) {
            double min_dist = 1e12;
            double tmp;
            for (int j = 0; j < k; j++) {
                if (j == i) {
                    continue;
                }
                tmp = dist(cents + i*f, cents + j*f, f);
                if (tmp < min_dist) {
                    min_dist = tmp;
                }
            }
            centers_min_dists[i] = min_dist;
        }

        // mapper calculate cid
        int samples_single_thread = n / threads;
        int left = n % threads;
        std::thread trainers[threads];
        for (int i = 0; i < threads; i++) {
            trainers[i] = std::thread(train_slave, i * samples_single_thread, (i + 1) * samples_single_thread + (i == threads - 1 ? left : 0),
            k, f, c, d, new_centers, m, cents, centers_min_dists, min_sim);
        }
        for (int i = 0; i < threads; i++) {
            trainers[i].join();
        }

        // reduce update centers
        int update = 0;
        int points_no_cls = 0;
        for (int i = 0; i < n; i++)
        {
            int oldcent = c[i];
            int newcent = new_centers[i];
            
            // count points_no_cls
            if (newcent == -1)
                points_no_cls += 1;

            // a = -1
            if (oldcent == -1) {
                if (newcent == -1)
                    continue;
                else {
                    update += 1;
                    centsC[newcent] += 1;
                    for (int j = 0; j < f; j++)
                    {
                        centsA[newcent * f + j] += m[i * f + j];
                    }
                    c[i] = newcent;
                }
            }
            // a != -1
            else {
                if (newcent == -1) {
                    update += 1;
                    centsC[oldcent] -= 1;
                    for (int j = 0; j < f; j++) {
                        centsA[oldcent * f + j] -= m[i * f + j];
                    }
                    c[i] = newcent;
                } else if (oldcent != newcent) {
                    update += 1;
                    centsC[oldcent] -= 1;
                    for (int j = 0; j < f; j++) {
                        centsA[oldcent * f + j] -= m[i * f + j];
                    }
                    centsC[newcent] += 1;
                    for (int j = 0; j < f; j++)
                    {
                        centsA[newcent * f + j] += m[i * f + j];
                    }
                    c[i] = newcent;
                } else {
                    continue;
                }
            }
        }

        for (int i = 0; i < k; i++)
        {
            // null cluster
            if (centsC[i] == 0) {
                for (int j = 0; j < f; j++)
                    cents[i * f + j] = 0.0;
                continue;
            }
            // cal mod
            double mod = 0.0;
            for (int j = 0; j < f; j++)
            {
                cents[i * f + j] = centsA[i * f + j] / centsC[i];
                mod += cents[i * f + j] * cents[i * f + j];
            }
            // remove mod
            if (mod != 0.0) {
                mod = sqrt(mod);
                for (int j = 0; j < f; j++) {
                    cents[i * f + j] = cents[i * f + j] / mod;
                }
            }
        }

        if (update <= n >> 8)
        {
            break;
        }
        niter += 1;
        fprintf(stderr, "kmeans iteration: %d, change instance : %d, points left : %d/%d\n", niter, update, points_no_cls, n);
    }

    for (int i = 0; i < n; i++) {
        if (c[i] != -1) {
            double dist_ = dist(m+i*f, cents+c[i]*f, f);
            scores[i] = (dist_*dist_ - 2) / -2;
        }
        else
            scores[i] = -1.0;
    }

    // save centers
    save_centers(k, f, cents, centsC, centers_path_save);

    free(cents);
    cents = NULL;
    free(centsA);
    centsA = NULL;
    free(centsC);
    centsC = NULL;

    return 0;
}
