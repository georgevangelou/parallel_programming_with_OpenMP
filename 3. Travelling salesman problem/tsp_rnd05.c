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
    Compiles/Runs/Debugs with: gcc tsp_rnd05.c -o tsp_rnd05 -lm -fopt-info -pg -fopenmp -O3 && time ./tsp_rnd05 && gprof ./tsp_rnd05
	Executes the algorithm for 10.000 cities, spanning in an area of 1.000x1.000 km and produces correct results
	Inherits all settings of the previous version unless stated otherwise
	Because of the fact that the use of rand() by a thread blocks other threads from executing it, each thread is given a pseudo-random list
		of numbers beforehand (<RandomSequences> array)
	Initializing the <RandomSequences> array takes a lot of time. Thus, a good speed improvement in the next version could be achieved if 
		one of the threads became responsible for refilling the array while the others perform the swappings
	Best time achieved: ~ 58 seconds to find a path below 500000 total distance with all optimizations listed below and 01 thread
		   				~ 27 seconds to find a path below 500000 total distance with all optimizations listed below and 12 threads
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

#define THREADS 12
#define STEPS_PER_THREAD_PER_REPETITION 20000000
#define DEFAULT_MAX_REPETITIONS 100000
#define THRESHOLD 500000

#define RANDOM_NUMBERS_PER_THREAD 2*STEPS_PER_THREAD_PER_REPETITION
#define STEPS_PER_THREAD_PER_REPETITION_2 2*STEPS_PER_THREAD_PER_REPETITION

// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
omp_lock_t Locks[N+1];
int RandomSequences[THREADS][RANDOM_NUMBERS_PER_THREAD];


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
// Finds Euclidean Distance between two cities
// ****************************************************************************************************************
double Distance(int A, int B) {
	double result = sqrt(   (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B]) );
	return result;
}


// ****************************************************************************************************************
// Finds Euclidean Distance in current path
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
// Swaps cities if swapping results in shorter Distance
// ****************************************************************************************************************
double SwapCities(double totDist) {
	double totDistChange = 0.0;

	#pragma omp parallel reduction(+:totDistChange) num_threads(THREADS)
	{
		int thisThreadsID = omp_get_thread_num();
		int r = 0;
		do {

			int A = RandomSequences[thisThreadsID][r++];
			int B = RandomSequences[thisThreadsID][r++];
			//if (r == RANDOM_NUMBERS_PER_THREAD+1) {printf("EXCEEDED STORED RANDOM NUMBERS"); exit(100);}

			if (A>B) { int temp = A; A = B; B = temp; } //always: A<B 
			int flag = B-A-1; //always:flag=0 when A+1==B

			omp_set_lock(&Locks[A]); omp_set_lock(&Locks[B]); 

			double dist1_old = Distance(Path[A-1], Path[A]); //is always needed
			double dist2_old = (!flag) ? 0 : Distance(Path[A], Path[A+1]); //dist ommited when A,B consecutive
			double dist3_old = (!flag) ? 0 : Distance(Path[B-1], Path[B]); //dist ommited when A,B consecutive
			double dist4_old = Distance(Path[B], Path[B+1]); //is always needed
			double dist1_new = Distance(Path[A-1], Path[B]); //is always needed
			double dist2_new = (!flag) ? 0 : Distance(Path[B], Path[A+1]); //dist ommited when A,B consecutive
			double dist3_new = (!flag) ? 0 : Distance(Path[B-1], Path[A]); //dist ommited when A,B consecutive
			double dist4_new = Distance(Path[A], Path[B+1]); //is always needed

			double distChange = - dist1_old - dist2_old - dist3_old - dist4_old + dist1_new + dist2_new + dist3_new + dist4_new; 
			if (distChange < 0) { //Must be <0 if it decreases the total Distance
				int temp = Path[A];
				Path[A] = Path[B];
				Path[B] = temp;
			} else distChange=0;
			omp_unset_lock(&Locks[A]); omp_unset_lock(&Locks[B]);
			totDistChange += distChange;
		} while (r < STEPS_PER_THREAD_PER_REPETITION_2) ;
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
// Initializes the locks
// ****************************************************************************************************************
void InitializeLocks() {
	for (int i=0; i<N+1; i++)
		omp_init_lock(&Locks[i]);
}


// ****************************************************************************************************************
// Initializes the random sequences array ensuring adjucent numbers are never the same
// ****************************************************************************************************************
void InitializeRandomSequencesArray(int rep) {
    int previous = -1, current;
    for (int i=0; i<THREADS; i++) {
		//srand(i*rep*rand());
        for (int j=0; j<RANDOM_NUMBERS_PER_THREAD; j++) {
            do{ 
                current = 1 + (int)(  ((float) rand())*(N-2)/((float)RAND_MAX)  );
            } while(current==previous);
            RandomSequences[i][j] = current;
            previous = current;
        }
	}
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
		printf("\nERROR: NOT ENOUGH SPACE ALLOCATED FOR GIVEN NUMBER OF CITIES\nThe program will now exit.\n"); return 1; }

	int repetitions = 0, MaxRepetitions = DEFAULT_MAX_REPETITIONS; omp_set_dynamic(0);
	if (argc>1) MaxRepetitions = atoi(argv[1]);

	printf("Maximum number of repetitions set at: %d\n", MaxRepetitions);
	printf("Maximum number of steps per thread per repetition set at: %d\n", MaxRepetitions*STEPS_PER_THREAD_PER_REPETITION);
    SetCities();
	ResetPath();
	RandomizePath();
	InitializeLocks();

	double prevDist, totDist = PathDistance();
	printf("Now running the main algorithm...\n");
	do {
		prevDist = totDist;
		repetitions ++;
        InitializeRandomSequencesArray(repetitions);
		//srand(repetitions*23);
		if (repetitions%1==0) printf(">>REPETITION:%8d  >>BATCH:%10d  >>ESTIMATED PATH_LENGTH: %.1lf\n", repetitions, repetitions*STEPS_PER_THREAD_PER_REPETITION*THREADS, totDist);	
		totDist = SwapCities(totDist);
	} while ((repetitions<MaxRepetitions) && (totDist>THRESHOLD));

	printf("\nCalculations completed. Results:\n");
	printf("Main-routine Repetitions: %d\n", repetitions);
	printf(" Sub-routine Repetitions: %d\n", repetitions*STEPS_PER_THREAD_PER_REPETITION*THREADS);
	//printf("Estimation of the optimal path length: %.2lf\n", totDist);
	printf("Actual optimal path length: %.2lf\n", PathDistance());
    return 0 ;
}






