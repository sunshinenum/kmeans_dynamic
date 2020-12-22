
#ifndef _KMEANS_H
#define _KMEANS_H

/* ------------------------------------------------
 * brief     : kmeans++
 * float * m : input data
 * int     n : number of rows   of input
 * int     f : number of column of input
 * int     k : cluster number
 * int   * c : cluster ID from 0 ~ k-1
 * return    : 0 success else failed
 * ------------------------------------------------ */
int kmeans(double *m, int n, int f, int k, int *c, char* old_centers_path, int old_centers_ct, int new_centers_keep, char* centers_path_save, double *scores, int threads, double min_sim);


#endif //KMEANS_H
