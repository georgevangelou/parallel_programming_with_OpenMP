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
    Compiles with: gcc kmeans03.c -o kmeans03 -lm (-O2)
    Uses indices to access array data
    Executes the algorithm for 10000 vectors of 100 dimensions and 10 classes
    Produces correct results
    Needs ~9 minutes to reach 16 repetitions without optimizations
          ~2 minutes to reach 16 repetitions with -O2 optimizations

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds.
    %   cumulative   self              self     total           
    time   seconds   seconds    calls   s/call   s/call  name    
    98.80  1230.92  1230.92       36    34.19    34.19   estimateClasses
    1.09   1244.56    13.64       36     0.38     0.38   estimateCenters
    0.06   1245.27     0.71        1     0.71     0.71   SetVec
    0.00   1245.27     0.00      100     0.00     0.00   notIn
    0.00   1245.27     0.00        1     0.00     0.00   initCenters 
*/


// ***************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ***************************************************
#define N  100000
#define Nv 1000
#define Nc 100
#define THRESHOLD 0.000001

// ***************************************************
double Vectors[N][Nv]; // N vectors of Nv dimensions
double Centers[Nc][Nv]; // Nc vectors of Nv dimensions
int   Class_of_Vec[N]; // Class of each Vector


// ***************************************************
// Print vectors
// ***************************************************
void printVectors(void) {
	int i, j;

	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Vector #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", Vectors[i][j]);
	}
}


// ***************************************************
// Print centers
// ***************************************************
void printCenters(void) {
	int i, j;

	for (i = 0; i < Nc; i++) {
		printf("--------------------\n");
		printf(" Center #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", Centers[i][j]);
	}
}

// ***************************************************
// Print the class of each vector
// ***************************************************
void printClasses(void) {
	int i, j;

	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Class of #%d is:\n", i);
		printf("  %d\n", Class_of_Vec[i]);
	}
}


// ***************************************************
// Check if a number is in a vector
// ***************************************************
int notIn(int Num, int Vec[Nc], int max_index){
	int j;
    for (j=0; j<max_index; j++)
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
        k =  (int) N * (1.0 * rand())/RAND_MAX ; // Pick a random integer in range [0, N-1]

        if ( notIn(k, Current_centers_indices, i) ) {
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
    double min_dist, dist, tot_min_distances = 0;
    int temp_class;
	int i, j, w;
	
    for (w=0; w<N; w++) {
        min_dist = 0;
		temp_class = 0;

        for (j=0; j<Nv; j++) 
            min_dist += (Vectors[w][j]-Centers[0][j]) * (Vectors[w][j]-Centers[0][j]); // Distance between Vec and Center 0
       
        for (i=1; i<Nc; i++) {
            dist = 0;
            
            for (j=0; j<Nv; j++) {
                dist = dist + (Vectors[w][j]-Centers[i][j]) * (Vectors[w][j]-Centers[i][j]); // Distance between Vec and Center i
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
    // Zero all center vectors
	for (i = 0; i < Nc; i++) {
		for (j = 0; j < Nv; j++) {
			Centers[i][j] = 0;
		}
	}

    // Add each vector's values to its corresponding center
    for (w = 0; w < N; w ++) {
        Centers_matchings[Class_of_Vec[w]] ++;
        for (j = 0; j<Nv; j++) {
            Centers[Class_of_Vec[w]][j] += Vectors[w][j];
        }
    }

	for (i = 0; i < Nc; i++) {
		if (Centers_matchings[i] != 0)
			for (j = 0; j < Nv; j++)
				Centers[i][j] /= Centers_matchings[i];
		else
			printf("\nERROR: CENTER %d HAS NO NEIGHBOURS...\n", i);
	}
}


// ***************************************************
// Initializing the vectors with random values
// ***************************************************
void SetVec( void ) {
    int i, j;

    for( i = 0 ; i< N ; i++ )
        for( j = 0 ; j < Nv ; j++ )
            Vectors[i][j] =  (1.0*rand())/RAND_MAX ;

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
