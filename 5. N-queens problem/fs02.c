/*
    Description:
        This program implements my backtracking method of solving the "N-Queens Problem"
        Abides by Lab 5 Exercise 1 requirements

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
        Compiles/Runs/Debugs with: gcc fs02.c -o fs02 -lm -fopt-info -pg && time ./fs02 && gprof ./fs02
        Inherits all features of previous version if not stated otherwise
        Removed X positional array, as each queen's X-position is determined by its index in the posY array
        Improved performance over previous version
        Still most time is consumed by isSafeFromPrevious()

        Profiler output for N=30 queens and no optimizations:
            Each sample counts as 0.01 seconds.
                %  cumulative     self                  self     total           
             time     seconds  seconds        calls  ms/call  ms/call  name    
            91.63       58.86    58.86   1712673314     0.00     0.00  isSafeFromPrevious
             7.20       63.49     4.62    114841495     0.00     0.00  SolveColumn
             1.26       64.30     0.81                                 main
             0.12       64.37     0.08            1    75.15    75.15  Map2

        Needs:  ~0m00,001s  and       232 steps for N=08 queens
                ~0m00,009s  and     20909 steps for N=16 queens
                ~0m00,124s  and    409633 steps for N=20 queens
                ~0m00,137s  and    812381 steps for N=26 queens
                ~0m03,130s  and   6142242 steps for N=28 queens
                ~1m07,097s  and 114841495 steps for N=30 queens
                ~0m16,473s  and 114841495 steps for N=30 queens and -O3
                ~0m15,446s  and 114841495 steps for N=30 queens and all optimizations listed below
*/	


// ****************************************************************************************************************    
//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
//#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
//#pragma GCC target("avx")  //Enable AVX


// ****************************************************************************************************************    
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"


// ****************************************************************************************************************    
#define N 30



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

    __time_t seconds;
    srand(seconds);

    int posY[N] = {0}; //Each queen's Y-position
    int yMin[N] = {0}; //Used to block already visited states

    printf("Now solving the problem. Please wait...\n");
    int i = 1, steps = 0;
    while(i<N) {
        steps++;
        if (steps%10000==0) printf("\rStep: %3d, Queen: %2d", steps, i);
        //printf("i=%d  yMin=%d\n",i, yMin[i]);
        if (!SolveColumn(posY, i, yMin[i])) {
            yMin[i--] = 0;
            yMin[i] = posY[i] + 1;
            if (yMin[i] == N) i-=1;
            if (i<0 || (i==0 && yMin[0] == N)) {
                printf("Couldn't find a solution\n");
                exit(0);
            }
        } else {
            i++;
        }
    }
    printf("\rStep: %3d, Queen: %2d", steps, i);
    Map2(posY, N); 
    bool allSafe = true;
    for (int i=0; i<N-1; i++) {
        if (!isSafeFromPrevious(posY, i, posY[i])) {
            printf("WRONG SOLUTION!!\n");
            allSafe = false;
        }
    }
    if (allSafe) printf("The solution was validated and is accepted.\n");

}
