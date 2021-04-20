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
    Compiles with: gcc kmeans06.c -o kmeans06 -lm -fopt-info-vec-optimized
    Uses pointers to access array data
    Uses vector processing for execution speed-up
    Executes the algorithm for 10000 vectors of 100 dimensions and 10 classes
    Produces correct results
    Needs ~30 seconds to reach 16 repetitions 

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds.
    %   cumulative   self              self     total           
    time   seconds   seconds    calls  Ts/call  Ts/call  name    
    97.94     45.20    45.20                             estimateClasses
    1.17     45.74     0.54                              estimateCenters
    1.00     46.20     0.46                              SetVec
*/
    

// ******************************************************************* 
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 

// ******************************************************************* 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ***************************************************
#define N  100000
#define Nv 1000
#define Nc 100
#define THRESHOLD 0.000001

// ***************************************************
float Vectors[N][Nv]; // N vectors of Nv dimensions
float Centers[Nc][Nv]; // Nc vectors of Nv dimensions
int   Class_of_Vec[N]; // Class of each Vector


// ***************************************************
// Print vectors
// ***************************************************
void printVectors(void) {
	int i, j;
    float *Vec = &Vectors[0][0];
	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Vector #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", *Vec++);
	}
}


// ***************************************************
// Print centers
// ***************************************************
void printCenters(void) {
	int i, j;
    float *Cent = &Centers[0][0];
	for (i = 0; i < Nc; i++) {
		printf("--------------------\n");
		printf(" Center #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", *Cent++);
	}
}

// ***************************************************
// Print the class of each vector
// ***************************************************
void printClasses(void) {
	int i, j;
    int *Cll = &Class_of_Vec[0];
	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Class of #%d is:\n", i);
		printf("  %d\n", *Cll++);
	}
}


// ***************************************************
// Check if a number is in a vector
// ***************************************************
int notIn(int Num, int Vec[Nc], int max_index){
	int j;
    for (j=0; j<max_index; j++)
        if (*Vec++ == Num) return 0;
    return 1;
}


// ***************************************************
// Choose random centers from available vectors
// ***************************************************
void initCenters( void ) {
    int i = 0, j = 0, k = 0;
    int Current_centers_indices[Nc] = {-1};  

    float *Cent = &Centers[0][0], *Vec = &Vectors[0][0];
    do {
        k =  (int) N * (1.0 * rand())/RAND_MAX ; // Pick a random integer in range [0, N-1]

        if ( notIn(k, Current_centers_indices, i) ) {
            Current_centers_indices[i] = k;
            for (j=0; j<Nv; j++) *Cent++ = *Vec++;
            i++;
        }
    } while(i<Nc);
}

// ***************************************************
// Returns the total squared minimum distance between all vectors and their closest center
// ***************************************************
float estimateClasses(void) {
    float min_dist, dist, tot_min_distances = 0;
    int temp_class;
	int i, j, w;
	
    float *Cent = &Centers[0][0], *Vec = &Vectors[0][0];
    float temp;
    for (w=0; w<N; w++) {
        min_dist = 0;
		temp_class = 0;

        Vec  = &Vectors[w][0];
        Cent = &Centers[0][0];
        for (j=0; j<Nv; j++) {
            temp = (*Vec++) - (*Cent++);
            min_dist += temp * temp; // Distance between Vec and Center 0
        }
       
        for (i=1; i<Nc; i++) {
            dist = 0;
            Vec  = &Vectors[w][0];
            Cent = &Centers[i][0];
            for (j=0; j<Nv; j++) {
                temp = (*Vec++) - (*Cent++);
                dist += temp * temp; // Distance between Vec and Center i
            }
            if (dist < min_dist) {
                temp_class = i;
                min_dist = dist;
            }
        }

        Class_of_Vec[w] = temp_class;
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

    float *Cent = &Centers[0][0], *Vec = &Vectors[0][0];
    // Zero all center vectors
	for (i = 0; i < Nc*Nv; i++) {
		*Cent++ = 0;
	}

    // Add each vector's values to its corresponding center
    for (w = 0; w < N; w ++) {
        Centers_matchings[Class_of_Vec[w]] ++;
        for (j = 0; j<Nv; j++) {
            Centers[Class_of_Vec[w]][j] += Vectors[w][j];
        }
    }

	for (i = 0; i < Nc; i++) {
		if (Centers_matchings[i] != 0) {
            Cent = &Centers[i][0];
			for (j = 0; j < Nv; j++)
				(*Cent++) /= Centers_matchings[i];
        }
		else
			printf("\nERROR: CENTER %d HAS NO NEIGHBOURS...\n", i);
	}
}


// ***************************************************
// Initializing the vectors with random values
// ***************************************************
void SetVec( void ) {
    int i, j;

    float *Vec = &Vectors[0][0];
    for( i = 0 ; i< N*Nv ; i++ )
        *Vec++ =  (1.0*rand())/RAND_MAX ;

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
    
    printf("Now initializing centers...\n");
    initCenters() ;

	//printf("The vectors were initialized with these values:\n");
	//printVectors();
    //printf("\nThe centers were initialized with these values:\n);
	//printCenters();

	totDist = 1.0e30;
    printf("Now running the main algorithm...\n\n");
    do {
        repetitions++; 
        prevDist = totDist ;
        
        totDist = estimateClasses() ;
        estimateCenters() ;
        diff = (prevDist-totDist)/totDist ;

        //printf("\nCurrent centers are:\n");
		//printCenters();
        
        printf(">> REPETITION: %3d  ||  ", repetitions);
        printf("DISTANCE IMPROVEMENT: %.8f \n", diff);
    } while( diff > THRESHOLD ) ;

    printf("\nProcess finished!\n");
	printf("\nTotal repetitions were: %d\n", repetitions);
    //printf("\nFinal centers are:\n");
    //printCenters() ;
	//printf("\nFinal classes are:\n");
	//printClasses() ;
    //printf("\n\nTotal distance is %f\n", totDist);
    return 0 ;
}

//**********************************************************************************************************
