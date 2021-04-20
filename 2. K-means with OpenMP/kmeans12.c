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
    Compiles with: gcc kmeans12.c -o kmeans12 -lm -fopt-info -fopenmp -O3
    Inherits all settings of the previous version unless stated otherwise
    Uses OpenMP for parallel thread execution
    Executes the algorithm for 100000 vectors of 1000 dimensions and 100 classes and produces correct results
    Needs ~122 seconds to reach 16 repetitions without unsafe and openmp
           ~19 seconds to reach 16 repetitions without openmp
            ~5 seconds to reach 16 repetitions with all optimizations and schedule(static, 1000)

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds.
    %   cumulative   self              self     total           
    time   seconds   seconds    calls  Ts/call  Ts/call  name    
    92.99     13.43    13.43                             frame_dummy
    4.09     14.02     0.59                             estimateCenters
    2.98     14.45     0.43                             SetVec

*/

// ******************************************************************* 
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX


// ******************************************************************* 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// ***************************************************
#define N  100000
#define Nv 1000
#define Nc 100
#define THRESHOLD 0.000001
#define MAX_REPETITIONS 16

// ***************************************************
float Vectors[N][Nv]; // N vectors of Nv dimensions
float Centers[Nc][Nv]; // Nc vectors of Nv dimensions
int   Class_of_Vec[N]; // Class of each Vector



// ***************************************************
// Print vectors
// ***************************************************
void printVectors(void) {
	int i, j;
    printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Vector #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", Vectors[i][j]);
	}
    printf("--------------------\n");
    printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}


// ***************************************************
// Print centers
// ***************************************************
void printCenters(void) {
	int i, j;
    printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
	for (i = 0; i < Nc; i++) {
		printf("--------------------\n");
		printf(" Center #%d is:\n", i);
		for (j = 0; j < Nv; j++)
			printf("  %f\n", Centers[i][j]);
	}
    printf("--------------------\n");
    printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}


// ***************************************************
// Print the class of each vector
// ***************************************************
void printClasses(void) {
	int i, j;
    printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
	for (i = 0; i < N; i++) {
		printf("--------------------\n");
		printf(" Class of Vector #%d is:\n", i);
		printf("  %d\n", Class_of_Vec[i]);
	}
    printf("--------------------\n");
    printf("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n");
}


// ****************************************************
// Returns 1 if a Vector is not in an array of vectors
// ****************************************************
int notVectorInCenters(float Vec[Nv], int maxIndex) {

    // Examining all the centers until <maxIndex>
    //printf("\nChecking if vec is in centers...\n");
    for (int c=0; c<maxIndex; c++) {
        //printf("> Checking center %d...\n", c);
        int flag = 1; 
        for (int i=0; i<Nv; i++) {
            //printf(">> Checking dim %d...\n", i);
            
            if (Vec[i] != Centers[c][i]) {
                //printf(">>> This dimension is different, so no need to keep checking this center.\n");
                flag = 0;
                break;
            }
        }
        if (flag)     // If <flag> remains equal to 1, then the vector <Vec> is equal to current examined center <c>
            return 0; // So <Vec> is unsuitable to become a new Center

    }

    return 1;
}


// ****************************************************
// Picks a new center when the last one has no neighbours
// ****************************************************
void pickSubstituteCenter(int indexOfCenterToChange){
    int currentVec = 0;

    // Searching for a vector that is not a center, so as to mark it as one
    printf("> Now searching for a substitute center...\n");
    do {
        printf(">> Now examining vec:%d\n", currentVec);
        if (notVectorInCenters(Vectors[currentVec], Nc)) {
            printf(">>> Current vec is not in existing centers\n");
            for (int i=0; i<Nv; i++) 
                Centers[indexOfCenterToChange][i] = Vectors[currentVec][i];  
                
            printf(">>> Substituted old center with current vector\n");
            return;    // If a substitute center is found, stop this function             
        }
            
        printf(">>> WARNING: If the center was substituted, this line must not be present\n");
        currentVec ++; // else contunue searching
    } while (currentVec<N);

    printf("\n");
    return;
}


// ****************************************************
// Chooses the first unique Nc vectors as class centers
// ****************************************************
void initCenters2() {
    int currentCenter=0, currentVec=0;
    do {
        if (notVectorInCenters(Vectors[currentVec], currentCenter)) {
            for (int i=0; i<Nv; i++) 
                Centers[currentCenter][i] = Vectors[currentVec][i];
            currentCenter ++;                
            }
        currentVec++;
    } while (currentCenter<Nc);
}


// *************************************************************************
// Returns the sum of distances between all vectors and their closest center
// *************************************************************************
float estimateClasses() {
    float tot_min_distances = 0;
	
    #pragma omp parallel for reduction(+:tot_min_distances) schedule(static, 1000)
    for (int w=0; w<N; w++) {
        float min_dist = 1e30;
		int temp_class = -1;
       
        for (int i=0; i<Nc; i++) {
            float dist = 0;
            
            #pragma omp simd reduction(+:dist) // If <reduction> is omitted, the compiler protects from math error and does not perform SIMD 
            for (int j=0; j<Nv; j++) 
                dist += (Vectors[w][j]-Centers[i][j]) * (Vectors[w][j]-Centers[i][j]); // Distance between Vec and Center i

            if (dist < min_dist) {
                temp_class = i;
                min_dist = dist;
            }
        }
        Class_of_Vec[w] = temp_class; // Update the current vector's class with the new one
        tot_min_distances += sqrt(min_dist); // Increase the sum of distances
    }
    return tot_min_distances;
}


// ***************************************************
// Find the new centers
// ***************************************************
void estimateCenters() {
    int Centers_matchings[Nc] = {0};    
    int needToRecalculateCenters = 0;

    // Zero all center vectors
	for (int i = 0; i < Nc; i++)
		for (int j = 0; j < Nv; j++)
			Centers[i][j] = 0;
	
    // Add each vector's values to its corresponding center
    for (int w = 0; w < N; w ++) {
        Centers_matchings[Class_of_Vec[w]] ++;
        for (int j = 0; j<Nv; j++)
            Centers[Class_of_Vec[w]][j] += Vectors[w][j];
    }

	for (int i = 0; i < Nc; i++) {
		if (Centers_matchings[i] != 0)
			for (int j = 0; j < Nv; j++)
				Centers[i][j] /= Centers_matchings[i];
		else {
			printf("\nWARNING: Center %d has no members.\n", i);
            pickSubstituteCenter(i);
            needToRecalculateCenters = 1;
            break;
        }
	}
    if (needToRecalculateCenters == 1) estimateCenters();
}


// ***************************************************
// Initializing the vectors with random values
// ***************************************************
void SetVec( void ) {
    for(int i = 0 ; i< N ; i++ )
        for(int j = 0 ; j< Nv ; j++ )
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
    initCenters2() ;

	//printf("\nThe vectors were initialized with these values:");
	//printVectors();
    //printf("\n\nThe centers were initialized with these values:");
	//printCenters();

	totDist = 1.0e30;
    printf("Now running the main algorithm...\n\n");
    do {
        repetitions++; 
        prevDist = totDist ;
        
        totDist = estimateClasses() ;
        estimateCenters() ;
        diff = (prevDist-totDist)/totDist ;

        //printf("\n\n\nNew centers are:");
		//printCenters();
        
        printf(">> REPETITION: %3d  ||  ", repetitions);
        printf("DISTANCE IMPROVEMENT: %.8f \n", diff);
    } while( (diff > THRESHOLD) && (repetitions < MAX_REPETITIONS) ) ;

    printf("\n\nProcess finished!\n");
	printf("Total repetitions were: %d\n", repetitions);

    /*
    printf("\n\nFinal centers are:");
    printCenters() ;
	printf("\n\nFinal classes are:");
	printClasses() ;
    //printf("\n\nTotal distance is %f\n", totDist); */
    return 0 ;
}

//**********************************************************************************************************
