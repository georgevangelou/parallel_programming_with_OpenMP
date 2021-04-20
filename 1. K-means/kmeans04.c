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

cat /proc/cpuinfo {
    processor       : 0
    vendor_id       : AuthenticAMD
    cpu family      : 23
    model           : 8
    model name      : AMD Ryzen 5 2600 Six-Core Processor
    stepping        : 2
    microcode       : 0x800820b
    cpu MHz         : 1374.877
    cache size      : 512 KB
    physical id     : 0
    siblings        : 12
    core id         : 0
    cpu cores       : 6
    apicid          : 0
    initial apicid  : 0
    fpu             : yes
    fpu_exception   : yes
    cpuid level     : 13
    wp              : yes
    flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb hw_pstate sme ssbd sev ibpb vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 xsaves clzero irperf xsaveerptr arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif overflow_recov succor smca
    bugs            : sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass
    bogomips        : 6786.50
    TLB size        : 2560 4K pages
    clflush size    : 64
    cache_alignment : 64
    address sizes   : 43 bits physical, 48 bits virtual
    power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]
    }
        
Version Notes:
    Compiles with: gcc kmeans04.c -o kmeans04 -lm -fopt-info-vec-optimized
    Uses indices to access array data
    Uses vector processing for execution speed-up
    Executes the algorithm for 10000 vectors of 100 dimensions and 10 classes
    Produces correct results
    Needs ~1 minute and 2 seconds to reach 16 repetitions

Profiler Output:
    Flat profile:
    Each sample counts as 0.01 seconds.
    %   cumulative   self              self     total           
    time   seconds   seconds    calls  Ts/call  Ts/call  name    
    98.60   138.00   138.00                              estimateClasses
    1.17    139.64     1.63                              estimateCenters
    0.33    140.10     0.46                              SetVec
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
