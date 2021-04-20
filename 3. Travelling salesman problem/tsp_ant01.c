/*
Description:
    This program is an initial definition of my implementation of the "Ant Colony" algorithm to solve the "Travelling Salesman Problem"

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
    Compiles/Runs/Debugs with: gcc tsp_ant01.c -o tsp_ant01 -lm -O3 -fopt-info -pg && time ./tsp_ant01 && gprof ./tsp_ant01    
	Runs too slow (cannot even test for 100.000 cities) but produces seemingly correct results (maybe better than any other previous solution method)
	Needs:	~08 seconds to find optimal path for N=50 cities for ANTS=100 and REPETITIONS=20 with all optimizations listed below

*/


// ****************************************************************************************************************    
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX



// **************************************************************************************************************** 
#include "stdio.h"
#include "stdlib.h"
#include "math.h"


// ****************************************************************************************************************
#define N  50
#define Nx 1000
#define Ny 1000
#define nonExist -999999

#define ALPHA 0.50 //0.50    // Affects pherormone dependency
#define BETA  0.10 //0.50  // Affects path length dependency
#define GAMMA 1.00
#define RHO   0.50 //0.50 
#define TAU_INITIAL_VALUE 0.50 //0.50
#define ANTS  100

#define REPETITIONS 20
#define DEBUG 0



// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
double CalculatedDistances[N][N];

double TauValues[N][N]; // Pherormone values between all city pairs

double DistanceTravelled[ANTS]; // The total length of each ant's path
int AntsPaths[ANTS][N+1]; // The paths of all ants



// ****************************************************************************************************************
// Checks if a city is already in ants <ant> path (until path[currentPathLength])
// ****************************************************************************************************************
int IsInAntsPath(int city, int currentPathLength, int ant) {
	for (int i=0; i<currentPathLength; i++)
		if (AntsPaths[ant][i] == city) return 1;
	return 0;
}


// ****************************************************************************************************************
// Prints an int array
// ****************************************************************************************************************
void PrintIntArray(int ARRAY[], const int SIZE) {
	for (int i=0; i<SIZE; i++) {
		printf("%3d  ", ARRAY[i]);
	}
	printf("\n");
}


// ****************************************************************************************************************
// Find min of an double array
// ****************************************************************************************************************
double MinOfDoubleArray(double ARRAY[], const int SIZE) {
	double min = INFINITY;
	for (int i=0; i<SIZE; i++)
		if (ARRAY[i] < min) 
			min = ARRAY[i];
	return min;
}


// ****************************************************************************************************************
// Prints the cities' positions
// ****************************************************************************************************************
void PrintCities() {
	printf("> The cities are:\n");
	for (int i=0; i<N; i++) {
		printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
	}
	printf("\n");
}


// ****************************************************************************************************************
// Prints the travelling sequence of given path
// ****************************************************************************************************************
void PrintPath_2(int Path[N+1]) {
	printf("> The path is:\n");
	for (int i=0; i<N+1; i++) {
		printf(">> %d ", Path[i]);
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
			Map[j][i] = (float) nonExist;


	//printf("Quantized coordinates are:\n");
	for (int c=0; c<N; c++) {
		int x = (int) CitiesX[c] ;
		int y = (int) CitiesY[c] ;
		//printf(" City:%d  y=%d and x=%d\n",c,y,x);
		if (Map[y][x] == nonExist) Map[y][x] = c;
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
// Finds Eucleidian distance in a given path
// ****************************************************************************************************************
double PathDistance_2(int Path[N+1]) {
	double totDist = 0.0;
	for (int i=0; i<N; i++) {
		totDist += Distance(Path[i], Path[i+1]);
	}
	totDist += Distance(Path[N], Path[0]);
	return totDist;
}


// ****************************************************************************************************************
// Calculates the possibility that an ant <ant> located at city I travels to city J
// ****************************************************************************************************************
double Possibility(int ant, int I, int J, int ant_path_length) {
	double summation = 0.0;
	for (int j=0; j<N; j++) 
		if (!IsInAntsPath(j, ant_path_length, ant)) 
			summation += pow(TauValues[I][j],ALPHA) * pow(CalculatedDistances[I][j],-BETA);

	if (isnan(summation)) {printf("\nFATAL ERROR: <summation> IS NAN\nTHE PROGRAM WILL BE TERMINATED.\n"); exit(1);}
	
	return (pow(TauValues[I][J],ALPHA) * pow(CalculatedDistances[I][J],-BETA) / summation);
}


// ****************************************************************************************************************
// Calculates new Tau of path between cities <I> and <J>
// ****************************************************************************************************************
double CalculateNewTau(int I, int J) {
	if (DEBUG>1) printf("Calculating new tau value between %d and %d...\n", I, J);
	double DeltaTau = 0.0;
	// For each ant
	for (int ant=0; ant<ANTS; ant++) {
		if (DEBUG>1) printf("\r> Progress: %.2f%%", 100*(ant+1)/((float)ANTS));
		// ...run through its path
		for (int city=0; city<N; city++) {
			// ...and check if path from <I> to <J> was used
			if ( ((AntsPaths[ant][city]==I)||(AntsPaths[ant][city]==J)) && ((AntsPaths[ant][city+1]==I)||(AntsPaths[ant][city+1]==J)) ) {
				DeltaTau += 1.0 / DistanceTravelled[ant];
				//printf("Delta tau is: %.10lf\n", DeltaTau);
				break; 
			}
		}
	} 
	if (DEBUG==1) printf("Delta tau is: %.10lf\n", DeltaTau);
	if (DEBUG>1) printf(" ===> Completed.\n");
 
	return ( ((1-RHO) * TauValues[I][J]) + (GAMMA*DeltaTau) );
}


// ****************************************************************************************************************
// Calculates new Tau values of all pairs of cities
// ****************************************************************************************************************
void CalculateNewTaus() {
	if (DEBUG) printf("Now calculating new tau values of all pairs of cities...\n");
	for (int i=0; i<N; i++) {
		if (DEBUG) printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
		for (int j=i+1; j<N; j++) {
			double newTau = CalculateNewTau(i, j);
            TauValues[i][j] = newTau;
            TauValues[j][i] = newTau;  
		}
	} 
	if (DEBUG) printf(" ===> Completed.\n");
}


// ****************************************************************************************************************
// Initializes the Tau values
// ****************************************************************************************************************
void InitializeTauValues() {
	printf("Now initializing the tau values...\n");
	for (int i=0; i<N; i++) {
		printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
		for (int j=0; j<N; j++) {
			TauValues[i][j] = TAU_INITIAL_VALUE; //(float) rand() / RAND_MAX;
		}
	}
	printf(" ===> Completed.\n");
}


// ****************************************************************************************************************
// Finds all Eucleidian distances between all pairs of cities
// ****************************************************************************************************************
void CalculateAllDistances() {
    printf("Now calculating distances between all pairs of cities...\n");
	for (int i=0; i<N; i++) {
        printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
        for (int j=i+1; j<N; j++) {
		    double temp = Distance(i, j);
            CalculatedDistances[i][j] = temp;
            CalculatedDistances[j][i] = temp;        
        }
	}
    printf(" ===> Completed.\n");
}


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
// Ant <ant> starts finding a path, starting from <starting_city>
// ****************************************************************************************************************
void AntRun(int ant, int starting_city) {
	if (DEBUG) printf("> ANT %d IS RUNNING...\n", ant);
	double totDist = 0.0;
    int visited_cities = 1, current_city = starting_city;

    AntsPaths[ant][0] = starting_city; 
	AntsPaths[ant][N] = starting_city;
    do {
        if (DEBUG) printf("\r>> Progress: %.2f%%", 100*(visited_cities+1)/((float) N) );

        double highest_decision_value = 0.0;
        int next_city = -1;

		// For every city set a decision value and choose the one with the highest
        for (int i=0; i<N; i++) {
			if (IsInAntsPath(i, visited_cities, ant)) continue; //...if we are trying to access current city or a visited one, go to next
			double random_number = 100000.0 * rand() / ((double) RAND_MAX); // random_number = (0, 1)
			double decision_value = random_number * Possibility(ant, current_city, i, visited_cities);
			if (DEBUG>1) printf(">>> test-city %d, random number: %.5lf, decision value: %.20lf\n", i, random_number, decision_value);
            if (decision_value > highest_decision_value) { 
                next_city = i;
				highest_decision_value = decision_value;
            } 
        }
        AntsPaths[ant][visited_cities++] = next_city; //Add decided city to current ant's path
        totDist += CalculatedDistances[current_city][next_city]; //...add the distance to it
        current_city = next_city; //...and make it the current city
    } while (visited_cities < N);

	totDist += CalculatedDistances[current_city][starting_city];
	DistanceTravelled[ant] = totDist;

    if (DEBUG) printf(" ===> Finished\n");
}


// ****************************************************************************************************************
// The main program
// ****************************************************************************************************************
int main( int argc, const char* argv[] ) {
	printf("------------------------------------------------------------------------------\n");
	printf("This program searches for the optimal traveling distance between %d cities,\n", N);
	printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
	printf("------------------------------------------------------------------------------\n");
    
    srand(1046900);
    SetCities();
    CalculateAllDistances();
	InitializeTauValues();

	int repetitions = 0;
	do {
		//printf("\n===============================================================\n");
		for (int ant=0; ant<ANTS; ant++) {
			int starting_city = (int) ((double)N*rand()/((double) RAND_MAX) );
			AntRun(ant, starting_city);
			if (DEBUG>1) printf(">> Ants path was:\n>>> ");
			if (DEBUG>1) PrintIntArray(AntsPaths[ant], N+1);
		}
		//printf("\n");
		CalculateNewTaus();
		//printf("-----------------------------------------------------------------\n");
		printf("REPETITION: %9d   ESTIMATED_OPTIMAL_PATH_LENGTH: %8.3lf\n", ++repetitions, MinOfDoubleArray(DistanceTravelled, ANTS));
		//printf("=================================================================\n\n");
	} while (repetitions < REPETITIONS);
	printf("\nCalculations completed. Results:\n");
	printf("Optimal path distance found is: %.2lf\n", MinOfDoubleArray(DistanceTravelled, ANTS));
	//printf("Estimated Ant0 distance found is: %.2lf\n", DistanceTravelled[0]);
	//printf("Real Ant0 distance found is: %.2lf\n", PathDistance_2(AntsPaths[0]));
    return 0 ;
}








