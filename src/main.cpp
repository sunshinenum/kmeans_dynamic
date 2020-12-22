
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmeans.h"
#include "npy.hpp"

void help()
{
    fprintf(stderr, "kmeans usage:\n");
    fprintf(stderr, "./kmeans -f [int dims] \n\
         -k [int clusters] \n\
         -d [string points_input] \n\
         -o [string output_dir] \n\
         -p [string old_centers_path] \n\
         -c [int old centers] \n\
         -n [int new centers keep] \n\
         -cp [string centers save path] \n\
         -threads [threads] \n\
         -min_sim [min sim]\n");
}

int main(int argc, char *argv[])
{
    if (argc != 21)
    {
        help();
        return -1;
    }
    srand(time(NULL));
    int f, k, n, i;
    char *old_centers_path = NULL;
    int old_centers, new_centers_keep;
    int *c = NULL;
    double *m = NULL;
    char *inf = NULL, *o = ".";
    char buffer[4096] = {0};
    char of[256] = {0};
    FILE *fp = NULL;
    f = atoi(argv[2]);
    k = atoi(argv[4]);
    inf = argv[6];
    o = argv[8];
    old_centers_path = argv[10];
    old_centers = atoi(argv[12]);
    new_centers_keep = atoi(argv[14]);
    char *centers_path_save = NULL;
    centers_path_save = argv[16];
    int threads = atoi(argv[18]);
    double min_sim = atof(argv[20]);

    // load points
    std::string path(inf);  
    std::vector<unsigned long> shape;
    std::vector<double> emb;
    std::cout << path << std::endl;
    npy::LoadArrayFromNumpy(path, shape, emb);
    n = shape[0];
    m = emb.data();
    c = (int *)calloc(sizeof(int), n);

    // train
    double *scores = (double *)malloc(sizeof(double) * n);
    kmeans(m, n, f, k, c, old_centers_path,
           old_centers, new_centers_keep, centers_path_save, scores, threads, min_sim);

    // save
    sprintf(of, "%s/clsid", o);

    fp = fopen(of, "w");
    if (fp == NULL)
    {
        fp = stdout;
    }

    i = 0;
    while (i < n)
    {
        fprintf(fp, "%d\t%lf\n", c[i], scores[i]);
        i++;
    }

    if (fp != stdout)
    {
        fclose(fp);
    }

    return 0;
}
