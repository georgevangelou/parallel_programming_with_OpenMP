/*
Description:
    This program executes my "Random Swapping" algorithm to solve the "Travelling Salesman Problem"

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
    Compiles/Runs/Debugs with: gcc tsp_rnd04.c -o tsp_rnd04 -lm -fopt-info -pg -fopenmp -O3 && time ./tsp_rnd04 && gprof ./tsp_rnd04
	Executes the algorithm for 10.000 cities, spanning in an area of 1.000x1.000 km and produces correct results
	Inherits all settings of the previous version unless stated otherwise
    Was used as a test to determine performance increment when using multiple threads. The use of rand() by multiple threads
	    made true parallel processing almost infeasible.

*/


// ****************************************************************************************************************    
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX


// **************************************************************************************************************** 
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"


// ****************************************************************************************************************
#define N 10000
#define Nx 1000
#define Ny 1000
#define VACANT_POSITION_CODE -999999
#define TOTAL_BATCHES 1e8
#define BATCH_SIZE 1200000
#define BATCH_SIZE_PER_RESCHEDULING 100000
#define DEFAULT_MAX_REPETITIONS TOTAL_BATCHES/BATCH_SIZE


// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
omp_lock_t Locks[N+1];



// ****************************************************************************************************************
// Initializes the cities' positions
// ****************************************************************************************************************
void SetCities() {
	printf("Now initializing the positions of the cities...\n");
	for (int i=0; i<N; i++) {
		CitiesX[i] = Nx * (float) rand() / RAND_MAX;
		CitiesY[i] = Ny * (float) rand() / RAND_MAX;
	}
}


// ****************************************************************************************************************
// Initializes the traveling path
// ****************************************************************************************************************
void ResetPath() {
	printf("Now initializing the path...\n");
	for (int i=0; i<N+1; i++)
		Path[i] = -1;
}


// ****************************************************************************************************************
// Checks if a city is already in the path
// ****************************************************************************************************************
int IsInPath(int k) {
	for (int i=0; i<N; i++)
		if (Path[i] == k) return 1;
	return 0;
}


// ****************************************************************************************************************
// Creates a random path
// ****************************************************************************************************************
void RandomizePath() {
	int k;
	printf("Now randomizing the path...\n");

	Path[0] = (N*rand())/RAND_MAX;
	Path[N] = Path[0];

	for (int i=1; i<N; i++) {
		
		do {
			k = ((float)N*rand())/RAND_MAX;
		} while (IsInPath(k) == 1);
		Path[i] = k;
	}
}


// ****************************************************************************************************************
// Prints the cities' positions
// ****************************************************************************************************************
void PrintCities() {
	int x, y;
	printf("> The cities are:\n");
	for (int i=0; i<N; i++) {
		printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
	}
	printf("\n");
}


// ****************************************************************************************************************
// Visually maps the cities' positions
// ****************************************************************************************************************
void MapCities() {
	int Map[Ny+1][Nx+1];
	printf("Now creating a visual map of the cities...\n");
	for (int i=0; i<Nx+1; i++) 
		for (int j=0; j<Ny+1; j++) 
			Map[j][i] = (float) VACANT_POSITION_CODE;


	//printf("Quantized coordinates are:\n");
	for (int c=0; c<N; c++) {
		int x = (int) CitiesX[c] ;
		int y = (int) CitiesY[c] ;
		//printf(" City:%d  y=%d and x=%d\n",c,y,x);
		if (Map[y][x] == VACANT_POSITION_CODE) Map[y][x] = c+1;
		else Map[y][x] = -1;
	}

	printf("This is the cities' map:\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	for (int y=0; y<Ny+1; y++){
		for (int x=0; x<Nx+1; x++)
			printf("%8d ", Map[y][x]);
		printf("\n");
	}
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("\n");
}


// ****************************************************************************************************************
// Finds Squared Euclidean Distance between two cities
// ****************************************************************************************************************
double Distance(int A, int B) {
	double result = sqrt(   (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B]) );
	//double result = (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B]) ;
	return result;
}


// ****************************************************************************************************************
// Finds Euclidean Distance in current path
// ****************************************************************************************************************
double PathDistance() {
	double totDist = 0.0;
	//#pragma omp simd reduction(+:totDist) //makes no difference
	//#pragma omp parallel for reduction(+:totDist) //slightly faster without it
	for (int i=0; i<N; i++) {
		totDist += Distance(Path[i], Path[i+1]);
	}
	totDist += Distance(Path[N], Path[0]);
	return totDist;
}


// ****************************************************************************************************************
// Swaps cities if swapping results in shorter Distance
// ****************************************************************************************************************
double SwapCities(double totDist) {
	double totDistChange = 0.0;

	#pragma omp parallel for reduction(+:totDistChange) schedule(static, BATCH_SIZE_PER_RESCHEDULING) //without this the program is vastly faster (single core)
	for (int counter=0; counter<BATCH_SIZE; counter++)
	//#pragma omp parallel reduction(+:distChange) //removing totDistChange
	{
		//srand((int) time(NULL) ^ omp_get_thread_num()); //severely hurts performance

		int A = (rand() %  (N-1 - 1 + 1)) + 1; //Picking a random index inside Path (0 < A < N)
		int B = (rand() %  (N-1 - 1 + 1)) + 1; //Picking a random index inside Path (0 < B < N)
		
		while (A==B) B = (rand() %  (N-1 - 1 + 1)) + 1; //If B==A, find another B

		if (A>B) { int temp = A; A = B; B = temp; } // So that A<B
		int flag = B-A-1; // Zero only when B==A+1

		double dist1_old, dist2_old, dist3_old, dist4_old, dist1_new=1, dist2_new, dist3_new, dist4_new;

		//#pragma omp parallel sections //Severely worsens performance
		dist1_old = Distance(Path[A-1], Path[A]); //is always needed
		dist2_old = (!flag) ? 0 : Distance(Path[A], Path[A+1]); //dist ommited when A,B consecutive
		dist3_old = (!flag) ? 0 : Distance(Path[B-1], Path[B]); //dist ommited when A,B consecutive
		dist4_old = Distance(Path[B], Path[B+1]); //is always needed
		dist1_new = Distance(Path[A-1], Path[B]); //is always needed
		dist2_new = (!flag) ? 0 : Distance(Path[B], Path[A+1]); //dist ommited when A,B consecutive
		dist3_new = (!flag) ? 0 : Distance(Path[B-1], Path[A]); //dist ommited when A,B consecutive
		dist4_new = Distance(Path[A], Path[B+1]); //is always needed

		double distChange = - dist1_old - dist2_old - dist3_old - dist4_old + dist1_new + dist2_new + dist3_new + dist4_new; 
		
		if (distChange < 0) { //Must be <0 if it decreases the total Distance
			//Setting the locks here is wrong, as any secondary thread trying to move a city will have calculated
			//wrong distances
			omp_set_lock(&Locks[A]); omp_set_lock(&Locks[B]); 
			int temp = Path[A];
			Path[A] = Path[B];
			Path[B] = temp;
			omp_unset_lock(&Locks[A]); omp_unset_lock(&Locks[B]);
		} else distChange=0;
		totDistChange += distChange;
	}
	return totDist + totDistChange;
}


// ****************************************************************************************************************
// Checks if current program parameters lead to feasible spacial states
// ****************************************************************************************************************
int ValidateParameters() {
	if (Nx*Ny<N) return 0;
	return 1;
}


// ****************************************************************************************************************
// Initializes locks
// ****************************************************************************************************************
void InitializeLocks() {
	for (int i=0; i<N+1; i++)
		omp_init_lock(&Locks[i]);
}


// ****************************************************************************************************************
// The main program
// ****************************************************************************************************************
int main( int argc, const char* argv[] ) {
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program searches for the optimal traveling Distance between %d cities,\n", N);
	printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    
	if (ValidateParameters() == 0) {
		printf("\nERROR: NOT ENOUGH SPACE ALLOCATED FOR GIVEN NUMBER OF CITIES\n");
		printf("The program will now exit.\n");
		return 1;
	}
	int repetitions = 0, MaxRepetitions = DEFAULT_MAX_REPETITIONS;
	if (argc>1) MaxRepetitions = atoi(argv[1]);

	printf("Maximum number of repetitions set at: %d\n", MaxRepetitions);
	printf("Maximum number of batches set at: %lf\n", TOTAL_BATCHES);
    SetCities();
	ResetPath();
	RandomizePath();
	InitializeLocks();

	double totDist = PathDistance();
	printf("Now running the main algorithm...\n");
	do {
		repetitions ++;
		if (repetitions%10==0) printf(">>REPETITION:%8d  >>BATCH:%10d  >>PATH_LENGTH: %.1lf\n", repetitions, repetitions*BATCH_SIZE, totDist);	
		totDist = SwapCities(totDist);
		
	} while (repetitions < MaxRepetitions);

	printf("\nCalculations completed. Results:\n");
	printf("Repetitions: %d\n", repetitions);
	printf("Batches: %d\n", repetitions*BATCH_SIZE);
	//printf("Estimation of the optimal path length: %.2lf\n", totDist);
	printf("Actual optimal path length: %.2lf\n", PathDistance());
    return 0 ;
}






