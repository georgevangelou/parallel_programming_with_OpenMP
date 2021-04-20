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
    Compiles with: gcc kmeans01.c -o kmeans01 -lm
    Executes the algorithm for 2 vectors of 2 dimensions and 2 classes
    Produces correct results

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds. (no time accumulated)
    %   cumulative   self              self     total           
    time   seconds   seconds    calls  Ts/call  Ts/call  name    
    0.00      0.00     0.00        4     0.00     0.00  printVec
    0.00      0.00     0.00        2     0.00     0.00  estimateCenters
    0.00      0.00     0.00        2     0.00     0.00  estimateClasses
    0.00      0.00     0.00        2     0.00     0.00  notIn
    0.00      0.00     0.00        1     0.00     0.00  SetVec
    0.00      0.00     0.00        1     0.00     0.00  initCenters
*/


// ***************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ***************************************************
#define N 2
#define Nv 2
#define Nc 2
#define THRESHOLD 0.000001


// ***************************************************
float Vectors[N][Nv]; // N vectors of Nv dimensions
float Centers[Nc][Nv]; // Nc vectors of Nv dimensions
int   Deviation_of_Vec_k[N]; //Distance of Vec k from its class
int   Class_of_Vec[N]; // Class of each Vector


// ***************************************************
// Check if a number is in a vector
// ***************************************************
int notIn(int Num, int Vec[Nc], int max_index){
	int j;
    for ( j=0; j<max_index; j++)
        if (Vec[j]==Num) return 0;
    return 1;
}


// ***************************************************
// Choose random centers from available vectors
// ***************************************************
void initCenters( void ) {
    int i = 0, j = 0, k = 0;
    int Current_centers_indices[Nc] = {-1};  

    while(i<Nc) {
        k =  Nc * (1.0 * rand())/RAND_MAX ; // Pick a random integer in range [0, Nc-1]

        if ( notIn(k, Current_centers_indices, i) ) {
            printf("Center %i will get the value of vector %d\n",i, k);
            Current_centers_indices[i] = k;
            for (j=0; j<Nv; j++) Centers[i][j] = Vectors[k][j];
            i++;
        }
    }
}

// ***************************************************
// Returns the total squared minimum distance between all vectors and their closest center
// ***************************************************
float estimateClasses(void) {
    float min_dist, dist, tot_min_distances = 0;
    int class = 0;
	int i, j, w;
    for (w=0; w<N; w++) {
        min_dist = 0; class = 0;

        for (j=0; j<Nv; j++) 
            min_dist += (Vectors[w][j]-Centers[0][j])*(Vectors[w][j]-Centers[0][j]); // Distance between Vec and Center 0
        
        for (i=1; i<Nc; i++) {
            dist = 0;
            for ( j=0; j<Nv; j++) {
                dist += (Vectors[w][j]-Centers[i][j])*(Vectors[w][j]-Centers[i][j]); // Distance between Vec and Center i
            }
            if (dist < min_dist) {
                class = i;
                min_dist = dist;
            }
        }
        printf("Vector %d has class %d \n", w, class);
        Class_of_Vec[w] = class;
        tot_min_distances += sqrt(min_dist);
    }
    return tot_min_distances;
}


// ***************************************************
// Find the new centers
// ***************************************************
void estimateCenters( void ) {
    int Centers_matchings[Nc] = {0};    
	int i, j, w;
    // Zero all center vectors
    for (i = 0; i<Nc; i++) 
        for (j = 0; j<Nv; j++) {
            Centers[i][j] = 0;
            //Centers_matchings[i] = 0;
        }

    // Add each vector's values to its corresponding center
    printf("\n");
    for (w = 0; w < N; w ++){
        Centers_matchings[Class_of_Vec[w]] += 1;
        printf("Class of vector %d is %d\n", w,Class_of_Vec[w]);
        for (j = 0; j<Nv; j++) {
            Centers[Class_of_Vec[w]][j] += Vectors[w][j];
        }
    }

    printf("\n Now printing the new centers:\n");
    for (i = 0; i<Nc; i++) {
        printf("\n The center %d will become:",i);
        for (j = 0; j<N; j++) {
            printf("\n  Value   %f  and the matchings is %d", Centers[i][j], Centers_matchings[i]);
            Centers[i][j] /= Centers_matchings[i];
        }
    }
    printf("\n");
}


// ***************************************************
// Initializing the vectors with random values
// ***************************************************
void SetVec( void ) {
    int i, j;

    for( i = 0 ; i< N ; i++ )
        for( j = 0 ; j < Nv ; j++ )
            Vectors[i][j] = (1.0*rand())/RAND_MAX ;
}



// ***************************************************
// Print all vectors
// ***************************************************
void printVec( float *Vecs, int number ) {
    int i, j ;

    for( i = 0 ; i< N ; i++ ) {
        printf("--------------------\n");
        printf(" Vector #%d is:\n", i);
        for( j = 0 ; j < number ; j++ )
            printf( "  %f\n", *Vecs++ );
    }
}


// ***************************************************
// The main program
// ***************************************************
int main() {
    float totDist, prevDist;
    printf("--------------------------------------------------------------------------------------------------\n");
	printf("This program executes the K-Means algorithm for random vectors of arbitrary number and dimensions.\n");
	printf("Current configuration has %d Vectors, %d Classes and %d Elements per vector.\n", N, Nc, Nv);
	printf("--------------------------------------------------------------------------------------------------\n");
    printf("Now initializing vectors...\n");
    SetVec() ;
    
    printf("\nThe vectors were initialized with these values:\n");
    printVec(Vectors[0], Nv) ;

    printf("\nNow initializing centers...\n");
    initCenters() ;

    printf("\nCurrent centers are:\n");
    printVec(Centers[0], Nc) ;

    totDist = 1.0e30 ;
    do {
        prevDist = totDist ;
        totDist = estimateClasses() ;
        estimateCenters() ;

        printf("\nCurrent centers are:\n");
        printVec(Centers[0], Nc) ;

    } while( (prevDist-totDist)/totDist  >  THRESHOLD ) ;

    printf("\nProcess finished!");
    printf("\nThe program will now terminate.\n");

    return 0 ;
}

//**********************************************************************************************************
