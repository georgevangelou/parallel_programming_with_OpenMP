/*
    Description:
        This program implements my backtracking method of solving the "N-Queens Problem" using multiple threads
        Abides by Lab 5 Exercise 2 requirements

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
        Compiles/Runs/Debugs with: gcc fs03.c -o fs03 -lm -fopenmp -fopt-info -pg && time ./fs03 && gprof ./fs03
        Inherits all features of previous version if not stated otherwise
        Comment-out lines 178,179 in case compiling fails
        Uses multiple threads to dramastically improve performance. Each thread searches for a solution in a different set 
            of first-column queen positions. When a thread finds a correct solution, the algorithm is terminated and
            the solution is printed on the screen

            Needs:  ~0m00,003s  and         69 steps for N=08 queens with all optimizations listed below and parallel-computing
                    ~0m00,003s  and         73 steps for N=16 queens with all optimizations listed below and parallel-computing
                    ~0m00,005s  and       7869 steps for N=20 queens with all optimizations listed below and parallel-computing
                    ~0m00,010s  and      70130 steps for N=26 queens with all optimizations listed below and parallel-computing
                    ~0m00,035s  and     999667 steps for N=28 queens with all optimizations listed below and parallel-computing
                    ~0m00,170s  and    5472083 steps for N=30 queens with all optimizations listed below and parallel-computing
                    ~0m15,274s  and  429111979 steps for N=35 queens with all optimizations listed below and parallel-computing

            For N=35 queens, all optimizations listed below and parallel programming it needs:
                real	0m15,274s
                user	2m39,584s
                sys	    0m01,173s

*/	


// ****************************************************************************************************************    
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX


// ****************************************************************************************************************    
#include "stdio.h"
#include "stdlib.h"
#include "omp.h"
#include "stdbool.h"


// ****************************************************************************************************************    
#define N 35



/**
 * Initializes positional array given
 */
void Initialization(int posY[N]) {
    for (int i=0; i<N; i++) {
        posY[i] = 0;
    }
}


/**
 * Prints a map of the queens until the M-th positioned queen
 */
void Map2(int posY[N], int M) {
    //for (int i=0; i<M; i++) printf("~~~~", i); 
    printf("\n\n========================\n"); 
    printf("---- SOLUTION FOUND ----\n");
    printf("========================\n");
    for (int i=0; i<N; i++) printf("---"); printf("---\n##|");
    for (int i=0; i<N; i++) printf("%2d ", i+1); printf("\n---");
    for (int i=0; i<N; i++) printf("---"); printf("\n");
    
    for (int y=0; y<N; y++) {
        printf("%2d| ", y+1);
        for (int x=0; x<N; x++) {
            bool flag = false;
            for (int i=0; i<M; i++) {
                if (i==x && posY[i]==y) {
                    flag = true;
                }
            }
            if (flag) printf("Q");
            else printf("~");
            printf("  ");
        }
        printf("\n");
    }
    for (int i=0; i<N; i++) printf("---"); printf("---\n\n");
}


/**
 * Checks if a position is safe
 */
bool isSafeFromPrevious(int posY[N], int x, int y) {
    int currentQueen = x;
    for (int oldQueen=0; oldQueen<currentQueen; oldQueen++) {
        //printf("      Checking %d %d and %d %d \n",posX[q],posY[q],x,y);
        if (oldQueen==x || posY[oldQueen]==y) return false; //If row/column is endangered
        else if (y==posY[oldQueen]+(currentQueen-oldQueen) || y==posY[oldQueen]-(currentQueen-oldQueen)) return false; //If diagonal is endangered
    }
    return true;
}


/**
 * Searches for a safe spot in a specified column
 */
bool SolveColumn(int posY[N], int current, int startingPoint) {
    int x = current;
    for (int y=startingPoint; y<N; y++) {
        if (isSafeFromPrevious(posY, x, y)) {
            posY[current] = y;
            return true;
        }
    }
    return false;
}


/**
 * The main program
 */
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements my backtracking method of solving the \"N-Queens Problem\".\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    printf("Now solving the problem. Please wait...\n");
    
    int maxQueen = 0, threadSolved = -1;
    long int finalSteps = 0, steps = 0;
    int finalPosY[N] = {0};

    #pragma omp parallel shared(finalSteps, finalPosY, threadSolved, maxQueen) default(none) reduction(+:steps)
    {
        int posY[N] = {0}; //Each queen's Y-position
        int yMin[N] = {0}; //Used to block already visited states
        int i = 1;
        bool canContinue = true;
        posY[0] = (omp_get_thread_num()*N+1)/omp_get_num_threads();    

        #pragma omp critical
        printf("> I am thread %2d and my first queen will be at position: [ 0 , %2d]\n", omp_get_thread_num(), posY[0]);  
        #pragma omp barrier

        while(threadSolved<0 && canContinue) {
            steps++;
            if (!SolveColumn(posY, i, yMin[i])) {
                yMin[i--] = 0;
                yMin[i] = posY[i] + 1;
                if (yMin[i] == N) i-=1;
                if (i<0 || (i==0 && yMin[0] == N)) {
                    //printf("Couldn't find a solution\n");
                    canContinue = false;
                }
            } else {
                i++;
                if (i==N-1) {
                    #pragma omp critical
                    {
                        threadSolved = omp_get_thread_num();
                        for (int w=0; w<N; w++) 
                            finalPosY[w] = posY[w];
                        finalSteps = steps;
                        maxQueen = i;
                    }
                } else if (i>maxQueen) maxQueen = i;
                
            }
            #pragma omp master
            printf("\r>> Current furthest queen reached is: %d", maxQueen);
        }
    }
    
    printf("\n\nAlgorithm finished.\n");
    printf("> Solution found by Thread #%d in %ld steps\n", threadSolved, finalSteps);
    printf("> Sum of all threads's steps: %ld\n", steps);
    Map2(finalPosY, N); 

    bool allSafe = true;
    for (int i=0; i<N-1; i++) {
        if (!isSafeFromPrevious(finalPosY, i, finalPosY[i])) {
            printf("WRONG SOLUTION!!\n");
            allSafe = false;
        }
    }
    if (allSafe) printf("The solution was validated and is accepted.\n");
    

}
