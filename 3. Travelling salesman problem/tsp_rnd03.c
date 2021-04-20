/*
Description:
    This program executes my "Random Swapping" algorithm to solve the "Travelling Salesman Problem"
	Abides by Lab 3 Exercise 1 requirements

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
    Compiles/Runs/Debugs with: gcc tsp_rnd03.c -o tsp_rnd03 -lm -fopt-info -pg -O3 && time ./tsp_rnd03 && gprof ./tsp_rnd03
	Executes the algorithm for 10.000 cities, spanning in an area of 1.000x1.000 km and produces correct results
	Inherits all settings of the previous version unless stated otherwise
	Similar to previous version, but is stripped of debugging conditions and messages. Thus, it is faster.
    Needs 	~11 seconds to reach 100.000.000 repetitions with no optimizations
	    	~04 seconds to reach 100.000.000 repetitions with -O3
		   	~04 seconds to reach 100.000.000 repetitions with all optimizations listed below
			~06 seconds to find a path of length 600.000 (3x-worse than tsp_rnd07)
*/


// ****************************************************************************************************************    
//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
//#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
//#pragma GCC target("avx")  //Enable AVX


// **************************************************************************************************************** 
#include "stdio.h"
#include "stdlib.h"
#include "math.h"



// ****************************************************************************************************************
#define N 10000
#define Nx 1000
#define Ny 1000
#define VACANT_POSITION_CODE -999999
#define DEFAULT_MAX_REPETITIONS 1e9
#define THRESHOLD_DISTANCE 600000


// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int Path[N+1];



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
// Finds Euclidean distance between two cities
// ****************************************************************************************************************
double Distance(int A, int B) {
	return (double) sqrt(   (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B])   );
}


// ****************************************************************************************************************
// Finds Eucleidian distance in current path
// ****************************************************************************************************************
double PathDistance() {
	double totDist = 0.0;
	for (int i=0; i<N; i++) {
		totDist += Distance(Path[i], Path[i+1]);
	}
	totDist += Distance(Path[N], Path[0]);
	return totDist;
}


// ****************************************************************************************************************
// Swaps cities if swapping results in shorter distance
// ****************************************************************************************************************
double SwapCities(double totDist) {
	int A = (rand() %  (N-1 - 1 + 1)) + 1; //Picking a random index inside Path (0 < A < N)
	int B = (rand() %  (N-1 - 1 + 1)) + 1; //Picking a random index inside Path (0 < B < N)
	
	while (A==B) B = (rand() %  (N-1 - 1 + 1)) + 1; //If B==A, find another B

	if (A>B) { int temp = A; A = B; B = temp; } // So that A<B
	int flag = B-A-1; // Zero only when B==A+1

	double dist1_old = Distance(Path[A-1], Path[A]); //is always needed
	double dist2_old = (!flag) ? 0 : Distance(Path[A], Path[A+1]); //dist ommited when A,B consecutive
	double dist3_old = (!flag) ? 0 : Distance(Path[B-1], Path[B]); //dist ommited when A,B consecutive
	double dist4_old = Distance(Path[B], Path[B+1]); //is always needed
	double dist1_new = Distance(Path[A-1], Path[B]); //is always needed
	double dist2_new = (!flag) ? 0 : Distance(Path[B], Path[A+1]); //dist ommited when A,B consecutive
	double dist3_new = (!flag) ? 0 : Distance(Path[B-1], Path[A]); //dist ommited when A,B consecutive
	double dist4_new = Distance(Path[A], Path[B+1]); //is always needed

	double newDist = totDist - dist1_old - dist2_old - dist3_old - dist4_old + dist1_new + dist2_new + dist3_new + dist4_new;

	if (newDist < totDist) {
		int temp = Path[A];
		Path[A] = Path[B];
		Path[B] = temp;
		return newDist;
	}
	return totDist;
}


// ****************************************************************************************************************
// Checks if current program parameters lead to feasible spacial states
// ****************************************************************************************************************
int ValidateParameters() {
	if (Nx*Ny<N) return 0;
	return 1;
}


// ****************************************************************************************************************
// The main program
// ****************************************************************************************************************
int main( int argc, const char* argv[] ) {
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program searches for the optimal traveling distance between %d cities,\n", N);
	printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    
	if (ValidateParameters() == 0) {
		printf("\nERROR: NOT ENOUGH SPACE ALLOCATED FOR GIVEN NUMBER OF CITIES\n");
		printf("The program will now exit.\n");
		return 1;
	}
	int repetitions = 0;
	int MaxRepetitions = DEFAULT_MAX_REPETITIONS;
	if (argc>1) MaxRepetitions = atoi(argv[1]);
	printf("Maximum number of repetitions set at: %d\n", MaxRepetitions);
    SetCities();
	ResetPath();
	RandomizePath();

	double totDist = PathDistance();
	printf("Now running the main algorithm...\n");
	do {
		repetitions ++;
		totDist = SwapCities(totDist);	
	} while ((repetitions < MaxRepetitions) && (totDist>THRESHOLD_DISTANCE) ) ;

	printf("\nCalculations completed. Results:\n");
	printf("Repetitions: %d\n", repetitions);
	printf("Estimation of the optimal path length: %.2lf\n", totDist);
	printf("Actual optimal path length: %.2lf\n", PathDistance());
    return 0 ;
}






