#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define NUM 100000000

typedef struct nn{
    int myrank;
    int total_rank;
}calPi_param;


void* calPi(void *param) {

    long long int count;
    //long long int *ret = &count;
    double x_val, y_val, val;

    calPi_param *pi_param = param;

    long long int begin = (pi_param->myrank)*(NUM/pi_param->total_rank);
    long long int end = (NUM/pi_param->total_rank)*(pi_param->myrank+1);

    count = 0;

    for(long long int i=begin;i<end;i++) {
        x_val = rand()/(double)(RAND_MAX+1);
        y_val = rand()/(double)(RAND_MAX+1);
        //printf("x_val: %f, y_val: %f\n", x_val, y_val);
        val = x_val*x_val+y_val*y_val;
        //printf("val: %f\n", val);
        if(val < 1) {
            count++;
        }
    }

    //printf("before return: %ld\n", *ret);
    

    return (void *)count;

}


int main() {
    double pi_val;

/****** pthread *************/

    int thread_count = 10;
    pthread_t *thread_handles;
    calPi_param *Pi_param_handles;
    //long long int *ret;
    long long int count;

    pthread_mutex_t mutex;

/****************************/

    thread_handles = (pthread_t*)malloc(thread_count*sizeof(pthread_t));
    Pi_param_handles = (calPi_param*)malloc(thread_count*sizeof(calPi_param));

    pthread_mutex_init(&mutex, NULL);

    for(int i=0;i<thread_count;i++) {
        Pi_param_handles[i].myrank = i;
        Pi_param_handles[i].total_rank = thread_count;
        pthread_create(&thread_handles[i], NULL, calPi, &Pi_param_handles[i]);
    }

    count = 0;
    for(int i=0;i<thread_count;i++) {
        void *ret;
        pthread_join(thread_handles[i], &ret);
        printf("ret: %ld\n", (int)ret);
        pthread_mutex_lock(&mutex);
        count += (int)ret;
        pthread_mutex_unlock(&mutex);
    }

    pi_val = 4*(1.0*count)/NUM;

    printf("pi_val: %f\n", pi_val);

    pthread_mutex_destroy(&mutex);

    free(Pi_param_handles);
    free(thread_handles);

}