/*
Description:
    This program executes my implementation of the "Heinritz Hsiao" algorithm to solve the "Travelling Salesman Problem"
	Next city in path is either the closest or second closest one, depending on the value of <PICK_CLOSEST_CITY_POSSIBILITY>
	Abides by Lab 3 Exercise 4 requirements

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
    Compiles/Runs/Debugs with: gcc tsp_hh05.c -o tsp_hh05 -lm -O3 -pg && time ./tsp_hh05 && gprof ./tsp_hh05
	Executes the algorithm for 10.000 cities, spanning in an area of 1.000x1.000 km and produces correct results
	Inherits all settings of version tsp_hh03, unless stated otherwise
	Now instead of using the function IsInPath(), which accounted for the vast majority of the execution time,
        I created a boolean array which shows which cities have been visited
	Results when:   PICK_CLOSEST_CITY_POSSIBILITY = 1.00 ===> Minimum total path distance:  89517.46
					PICK_CLOSEST_CITY_POSSIBILITY = 0.95 ===> Minimum total path distance:  92572.41
					PICK_CLOSEST_CITY_POSSIBILITY = 0.90 ===> Minimum total path distance:  95408.65
					PICK_CLOSEST_CITY_POSSIBILITY = 0.85 ===> Minimum total path distance:  99890.06
					PICK_CLOSEST_CITY_POSSIBILITY = 0.80 ===> Minimum total path distance: 103822.38
					PICK_CLOSEST_CITY_POSSIBILITY = 0.75 ===> Minimum total path distance: 106003.44
    Needs:	~ 0.7 seconds to calculate the shortest steps path using 01 thread and all optimizations listed below 

*/


// **************************************************************************************************************** 
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX    


// **************************************************************************************************************** 
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"



// ****************************************************************************************************************
#define N  10000
#define Nx 1000
#define Ny 1000
#define nonExist -999999
#define PICK_CLOSEST_CITY_POSSIBILITY 0.75





// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
double CalculatedDistances[N][N];
bool CityIsVisited[N];



// ****************************************************************************************************************
// Initializes the <CityIsVisited> array
// ****************************************************************************************************************
void InitializeVisits() {
	printf("Now initializing the visited cities array...\n");
	for (int i=0; i<N; i++) {
		CityIsVisited[i] = false;
	}
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
    printf("Now calculating distances between all pairs of cities...\n");
	for (int i=0; i<N; i++) {
        printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
        for (int j=i+1; j<N; j++) {
		    double temp = Distance(i, j);
            CalculatedDistances[i][j] = temp;
            CalculatedDistances[j][i] = temp;        
        }
	}
    printf("\r> Progress: 100.00%% ===> Completed.\n");
}


// ****************************************************************************************************************
// Finds the travelling path by visiting the closest or second closest non-visited city each time
// ****************************************************************************************************************
double FindShortestStepPath_2() {
    printf("Now finding the shortest / second shortest step path...\n");
    double totDist = 0.0;
    int visited_cities = 1, current_city = 0;
    Path[0] = current_city; Path[N] = current_city; CityIsVisited[current_city] = true;
    do {
        printf("\r> Progress: %.2f%%", 100*(visited_cities+1)/((float)N));
        double dist = 0, min_dist_1 = INFINITY, min_dist_2 = INFINITY; 
        int closest_city_1 = -1, closest_city_2 = -1;
        for (int i=0; i<N; i++) {
            if (CityIsVisited[i]) continue; //If we are trying to access current city or a visited one, go to next
            dist = CalculatedDistances[current_city][i];
            if (min_dist_1 > dist) {
				min_dist_2 = min_dist_1; closest_city_2 = closest_city_1;
                min_dist_1 = dist; closest_city_1 = i;
            } else if (min_dist_2 > dist) {
				min_dist_2 = dist; closest_city_2 = i;
			}
        }
		int city_pick = ( (rand()/((float) RAND_MAX)) < PICK_CLOSEST_CITY_POSSIBILITY ) ? 1 : 2;
		int next_city =  (city_pick==1) ? closest_city_1 : closest_city_2;
        CityIsVisited[next_city] = true;
        Path[visited_cities++] = next_city; 
		current_city = next_city;
        totDist += (city_pick==1) ? min_dist_1 : min_dist_2;; 
    } while (visited_cities<N);
    printf(" ===> Completed.\n");
    totDist += CalculatedDistances[Path[N-1]][0];
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
    
    srand(1046900);
    SetCities();
    InitializeVisits();
    CalculateAllDistances();
    double totDistEstimation = FindShortestStepPath_2();
    printf("\n");
    printf("Estimated Total path distance is: %.2lf\n", totDistEstimation);
	printf("Exact Total path distance is: %.2lf\n", PathDistance());
    return 0 ;
}






