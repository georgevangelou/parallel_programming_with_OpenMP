/*
Description:
    This program executes my implementation of the "Heinritz Hsiao" algorithm to solve the "Travelling Salesman Problem"
	Next city in path is either the closest or second closest one, depending on the value of <PICK_CLOSEST_CITY_POSSIBILITY>

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
    Compiles/Runs/Debugs with: gcc tsp_hh04.c -o tsp_hh04 -lm -O3 -pg -fopenmp && time ./tsp_hh04 && gprof ./tsp_hh04
    Executes the algorithm for 10.000 cities, spanning in an area of 1.000x1.000 km and produces correct results
	Inherits all settings of the previous version unless stated otherwise
    Uses all available threads and each one finds a different path. The shortest one wins and its distance is displayed
    The use of rand() by every thread is the main reason that there is a decrease in execution time duration
	Function IsInPath_3() consumes the vast majority of the execution time
	Results when:   PICK_CLOSEST_CITY_POSSIBILITY = 1.00 ===> Minimum total path distance: 89517.46
					PICK_CLOSEST_CITY_POSSIBILITY = 0.95 ===> Minimum total path distance:
					PICK_CLOSEST_CITY_POSSIBILITY = 0.90 ===> Minimum total path distance: 
                    PICK_CLOSEST_CITY_POSSIBILITY = 0.85 ===> Minimum total path distance: 
                    PICK_CLOSEST_CITY_POSSIBILITY = 0.80 ===> Minimum total path distance: 
                    PICK_CLOSEST_CITY_POSSIBILITY = 0.75 ===> Minimum total path distance:
	Needs:	~ 02 minutes 00 seconds to calculate an optimal path using 12 threads and all optimizations listed below

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
#define N  10000
#define Nx 1000
#define Ny 1000
#define nonExist -999999
#define PICK_CLOSEST_CITY_POSSIBILITY 0.8
#define THREADS 12


// ****************************************************************************************************************
float CitiesX[N];
float CitiesY[N];
int ThreadsPath[THREADS][N+1];
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
int IsInPath_3(int city, int currentPathLength, int Path[]) {
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
void PrintPath_2(int Path[]) {
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
double PathDistance_2(int Path[]) {
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
    printf(" ===> Completed.\n");
}


// ****************************************************************************************************************
// Finds the travelling path by visiting the closest or second closest non-visited city each time
// ****************************************************************************************************************
double FindShortestStepPath_2() {
    #pragma omp master
    {
        printf("Now finding the shortest / second shortest step path...\n");
        printf("> Threads running independently in parallel: %d\n", omp_get_num_threads());
    }
    double totDist = 0.0;
    int visited_cities = 1, current_city = 0, thread = omp_get_thread_num();
	int foo = thread*omp_get_wtime()*523;
    ThreadsPath[thread][0] = 0; ThreadsPath[thread][N] = 0;
    do {
        #pragma omp master
        printf("\r> Progress: %.2f%%", 100*(visited_cities)/((float)N));
        double dist = 0, min_dist_1 = INFINITY, min_dist_2 = INFINITY; 
        int closest_city_1 = -1, closest_city_2 = -1;
        for (int i=0; i<N; i++) {
            if (IsInPath_3(i, visited_cities, ThreadsPath[thread])) continue; //If we are trying to access current city or a visited one, go to next
            dist = CalculatedDistances[current_city][i];
            if (min_dist_1 > dist) {
				min_dist_2 = min_dist_1; closest_city_2 = closest_city_1;
                min_dist_1 = dist; closest_city_1 = i;
            } else if (min_dist_2 > dist) {
				min_dist_2 = dist; closest_city_2 = i;
			}
        }
		unsigned seed = 11*visited_cities + 83*thread + 11*omp_get_wtime() + (foo++);
        float random_number = ((float)rand_r(&seed)) / ((float)RAND_MAX) ;
		int city_pick = (random_number<PICK_CLOSEST_CITY_POSSIBILITY) ? 1 : 2;

		int next_city =  (city_pick==1) ? closest_city_1 : closest_city_2;
        ThreadsPath[thread][visited_cities++] = next_city; 
		current_city = next_city;
        totDist += (city_pick==1) ? min_dist_1 : min_dist_2;; 
    } while (visited_cities<N);
    totDist += CalculatedDistances[ThreadsPath[thread][N-1]][0];
    #pragma omp barrier
    #pragma omp single
        printf("\r> Progress: 100.00%% ===> Completed.\n");
    #pragma omp barrier
    printf(">> I am thread #(%2d) and my total path distance is: %lf.02\n", thread, totDist);
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
    CalculateAllDistances();

    double totDistEstimation = INFINITY;
    #pragma omp parallel reduction(min:totDistEstimation) num_threads(THREADS)
    {
        totDistEstimation = FindShortestStepPath_2();
    }
    printf("\n");
    printf("Minimum total path distance found is: %.2lf\n", totDistEstimation);
    return 0 ;
}






