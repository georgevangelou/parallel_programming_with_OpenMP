/*
Description:
    This program is an initial definition of my implementation of the "Heinritz Hsiao" algorithm to solve the "Travelling Salesman Problem"

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
    Compiles/Runs/Debugs with: gcc tsp_hh01.c -o tsp_hh01 -lm -pg && time ./tsp_hh01 && gprof ./tsp_hh01
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



// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
double CalculatedDistances[N][N];


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
// Checks if a city is already in the path (until path[currentPathLength])
// ****************************************************************************************************************
int IsInPath2(int city, int currentPathLength) {
	for (int i=0; i<currentPathLength; i++)
		if (Path[i] == city) return 1;
	return 0;
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
// Prints the travelling path
// ****************************************************************************************************************
void PrintPath() {
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
// Finds all Eucleidian distances between all pairs of cities
// ****************************************************************************************************************
void CalculateAllDistances() {
    printf("Now calculating distances between all pairs of cities...");
	for (int i=0; i<N; i++) {
        for (int j=i+1; j<N; j++) {
		    double temp = Distance(i, j);
            CalculatedDistances[i][j] = temp;
            CalculatedDistances[j][i] = temp;        
        }
	}
}


// ****************************************************************************************************************
// Finds the travelling path by visiting the closest non-visited city each time
// ****************************************************************************************************************
double FindShortestStepPath() {
    double totDist = 0.0;
    int visited_cities = 1, current_city = 0;
    Path[0] = 0; Path[N] = 0;
    do {
        double dist = 0, min = INFINITY; 
        int next_city = -1;
        for (int i=0; i<N; i++) {
            printf("\ni=%d and currentCity=%d", i, current_city);

            if ( IsInPath2(i, visited_cities) ) {
                printf(" ===> ABORTED");
                continue; //If we are trying to access current city or a visited one, go to next
            }
            
            dist = CalculatedDistances[current_city][i];
            if (min > dist) {
                min = dist;
                next_city = i;
            }
        }
        Path[visited_cities++] = next_city;
        totDist += dist;
        printf("\nWent from %d to %d with distance %.2lf, so total distance is %.2lf", current_city, next_city, dist, totDist);
        current_city = next_city;
        
    } while (visited_cities<N);
    totDist += CalculatedDistances[Path[N-1]][0];
    printf("\nWent from %d to %d with distance %.2lf, so total distance is %.2lf", Path[N-1], 0, CalculatedDistances[Path[N-1]][0], totDist);
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
    PrintCities();
	MapCities();
    CalculateAllDistances();
    double totDistEstimation = FindShortestStepPath();
    printf("\n");
    printf("Estimated Total path distance is:%.2lf\n", totDistEstimation);
	printf("Exact Total path distance is:%.2lf\n", PathDistance());
    PrintPath();
    return 0 ;
}






