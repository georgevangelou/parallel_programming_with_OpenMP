/*
Description:
    This program is an initial definition of my "Random Swapping" algorithm to solve the "Travelling Salesman Problem"

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
    Compiles/Runs/Debugs with: gcc tsp_rnd01.c -o tsp_rnd01 -lm -pg && time ./tsp_rnd01 && gprof ./tsp_rnd01
	Initializes all required arrays 
	Prints the coordinates of the cities and creates a visual map representation to test correct initialization
	Contains manually tested implementations of basic functions
*/
    


// **************************************************************************************************************** 
#include "stdio.h"
#include "stdlib.h"
#include "math.h"



// ****************************************************************************************************************
#define N 3
#define Nx 7
#define Ny 7
#define nonExist -999999
#define MaxRepetitions 1e6



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

	printf("Random path generated is:\n");
	printf("> Destination=  0   City=%3d\n", Path[0]);
	for (int i=1; i<N; i++) {
		
		do {
			k = ((float)N*rand())/RAND_MAX;
		} while (IsInPath(k) == 1);
		Path[i] = k;
		printf("> Destination=%3d   City=%3d\n", i, Path[i]);
	}
	printf("> Destination=%3d   City=%3d\n", N, Path[N]);
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
			Map[j][i] = (float) nonExist;


	//printf("Quantized coordinates are:\n");
	for (int c=0; c<N; c++) {
		int x = (int) CitiesX[c] ;
		int y = (int) CitiesY[c] ;
		//printf(" City:%d  y=%d and x=%d\n",c,y,x);
		if (Map[y][x] == nonExist) Map[y][x] = c+1;
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
// Swaps cities if swapping results in shorter distance
// ****************************************************************************************************************
void SwapCities() {
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
// The main program
// ****************************************************************************************************************
int main( int argc, const char* argv[] ) {
	printf("------------------------------------------------------------------------------\n");
	printf("This program searches for the optimal traveling distance between %d cities,\n", N);
	printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
	printf("------------------------------------------------------------------------------\n");
    
    SetCities();
	ResetPath();
    PrintCities();
	MapCities();
	RandomizePath();
	printf("Total path distance is:%.2lf\n", PathDistance());
    
    return 0 ;
}






