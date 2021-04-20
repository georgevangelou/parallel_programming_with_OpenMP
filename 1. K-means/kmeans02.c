/*
Description:
    This program executes the K-Means algorithm for random vectors of arbitrary number
    and dimensions 

Author:
    Georgios Evangelou (1046900)
    Year: 5
    Parallel Programming in Machine Learning Problems
    Electrical and Computer Engineering Department, University of Patras

System Specifications:
    CPU: AMD Ryzen 2600  (6 cores/12 threads,  @3.8 GHz,  6786.23 bogomips)
    GPU: Nvidia GTX 1050 (dual-fan, overclocked)
    RAM: 8GB (dual-channel, @2666 MHz)
    
Version Notes:
    Compiles with: gcc kmeans02.c -o kmeans02 -lm
    Executes the algorithm for 10000 vectors of 100 dimensions and 10 classes
    Does not produce correct results

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds.
    %   cumulative   self              self     total           
    time   seconds   seconds    calls  ms/call  ms/call  name    
    91.02     0.10     0.10        2    50.06    50.06   estimateClasses
    9.10      0.11     0.01        2     5.01     5.01   estimateCenters
    0.00      0.11     0.00       10     0.00     0.00   notIn
    0.00      0.11     0.00        5     0.00     0.00   printVec
    0.00      0.11     0.00        1     0.00     0.00   SetVec
    0.00      0.11     0.00        1     0.00     0.00   initCenters
*/


// ***************************************************
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

// ***************************************************
#define N  10000
#define Nv 100
#define Nc 10
#define THRESHOLD 0.000001
#define REFACTORING 1

// ***************************************************
float Vectors[N][Nv]; // N vectors of Nv dimensions
float Centers[Nc][Nv]; // Nc vectors of Nv dimensions
int   Deviation_of_Vec_k[N]; //Distance of Vec k from its class
int   Class_of_Vec[N]; // Class of each Vector


// ***************************************************
// Check if a number is in a vector
// ***************************************************
int notIn(int Num, int Vec[Nc], int max_index){
    for (int j=0; j<max_index; j++)
        if (Vec[j]==Num) return 0;
    return 1;
}


// ***************************************************
// Choose random centers from available vectors
// ***************************************************
void initCenters( void ) {
    int i = 0, k = 0;
    int Current_centers_indices[Nc] = {-1};  

    while(i<Nc) {
        k =  N * (1.0 * rand())/RAND_MAX ; // Pick a random integer in range [0, N-1]

        if ( notIn(k, Current_centers_indices, i) ) {
            Current_centers_indices[i] = k;
            for (int j=0; j<Nv; j++) Centers[i][j] = Vectors[k][j];
            i++;
        }
    }
}

// ***************************************************
// Returns the total squared minimum distance between all vectors and their closest center
// ***************************************************
float estimateClasses(void) {
    float min_dist, dist, tot_min_distances = 0;
    int class = -1;

    for (int w=0; w<N; w++) {
        min_dist = 0; class = -1;

        for (int j=0; j<Nv; j++) 
            min_dist += REFACTORING * (Vectors[w][j]-Centers[0][j]) * (Vectors[w][j]-Centers[0][j]); // Distance between Vec and Center 0
        
        for (int i=1; i<Nc; i++) {
            dist = 0;
            
            for (int j=0; j<Nv; j++) {
                dist += REFACTORING * (Vectors[w][j]-Centers[i][j])*(Vectors[w][j]-Centers[i][j]); // Distance between Vec and Center i
                if (dist>min_dist) break;
            }
            //printf("\ndist is: %f", dist);
            if (dist < min_dist) {
                class = i;
                min_dist = dist;
            }
        }
        Class_of_Vec[w] = class;
        tot_min_distances += sqrt(min_dist);
    }
    //return pow(tot_min_distances, 0.5);
    return tot_min_distances;
    }


// ***************************************************
// Find the new centers
// ***************************************************
void estimateCenters( void ) {
    int Centers_matchings[Nc] = {0};    

    // Zero all center vectors
    for (int i = 0; i<Nc; i++) 
        for (int j = 0; j<Nv; j++) {
            Centers[i][j] = 0;
            //Centers_matchings[i] = 0;
        }

    // Add each vector's values to its corresponding center
    for (int w = 0; w < N; w ++){
        Centers_matchings[Class_of_Vec[w]] += 1;
        for (int j = 0; j<Nv; j++) {
            Centers[Class_of_Vec[w]][j] += Vectors[w][j];
        }
    }

    for (int i = 0; i<Nc; i++) {
        for (int j = 0; j<N; j++) {
            Centers[i][j] /= Centers_matchings[i];
        
        }
    }
}


// ***************************************************
// Initializing the vectors with random values
// ***************************************************
void SetVec( void ) {
    int i, j;

    for( i = 0 ; i< N ; i++ )
        for( j = 0 ; j < Nv ; j++ )
            Vectors[i][j] = REFACTORING * (1.0*rand())/RAND_MAX ;
}



// ***************************************************
// Print all vectors
// ***************************************************
void printVec( float *Vecs, int number ) {
    int i, j ;

    for( i = 0 ; i< number ; i++ ) {
        printf("--------------------\n");
        printf(" Vector #%d is:\n", i);
        for( j = 0 ; j < Nv ; j++ )
            printf( "  %f\n", *Vecs++ );
    }
}


// ***************************************************
// The main program
// ***************************************************
int main( int argc, const char* argv[] ) {
    int repetitions = 0;
    float totDist, prevDist, diff;

    printf("--------------------------------------------------------------------------------------------------\n");
	printf("This program executes the K-Means algorithm for random vectors of arbitrary number and dimensions.\n");
	printf("Current configuration has %d Vectors, %d Classes and %d Elements per vector.\n", N, Nc, Nv);
	printf("--------------------------------------------------------------------------------------------------\n");
    printf("Now initializing vectors...\n");
    SetVec() ;
    
    printf("The vectors were initialized with these values:\n");
    printVec(Vectors[0], Nv) ;
    printf("Now initializing centers...\n");
    initCenters() ;
    printf("\nCurrent centers are:\n");
    printVec(Centers[0], Nc) ;
    totDist = 1.0e30 ;

    printf("Now running the main algorithm...\n\n");
    do {
        repetitions++; 
        prevDist = totDist ;
        totDist = estimateClasses() ;
        estimateCenters() ;
        diff = (prevDist-totDist)/totDist ;
        printf("\nCurrent centers are:\n");
        printVec(Centers[0], Nc) ;
        
        printf(">> REPETITION: %3d  ||  ", repetitions);
        printf("DISTANCE IMPROVEMENT: %.8f \n", diff);
    } while(  diff  >  THRESHOLD ) ;

    printf("\nProcess finished!\n");
    scanf("%f", &prevDist);
    printf("\nFinal centers are:\n");
    printVec(Centers[0], Nc) ;
    printf("\nTotal repetitions were: %d",repetitions);
    printf("\n\nFinal distance is %f\n", totDist);
    return 0 ;
}

//**********************************************************************************************************