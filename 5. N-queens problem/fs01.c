/*
    Description:
        This program implements my backtracking method of solving the "N-Queens Problem"
        

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
        Compiles/Runs/Debugs with: gcc fs01.c -o fs01 -lm -fopt-info -pg && time ./fs01 && gprof ./fs01
        Correctly solves the problem
        Most time consumed by isSafeFromPrevious()

        Profiler output for N=30 queens and no optimizations:
            Each sample counts as 0.01 seconds.
                %  cumulative     self                   self    total           
             time     seconds   seconds        calls  ms/call  ms/call  name    
            91.79       76.84    76.84    1712673314     0.00     0.00  isSafeFromPrevious
             7.37       83.01     6.17     114841495     0.00     0.00  SolveColumn
             0.97       83.83     0.81                                  main
             0.05       83.87     0.05             1    45.08    45.08  Map

        Needs:  ~0m00,001s  and       232 steps for N=08 queens
                ~0m00,011s  and     20909 steps for N=16 queens
                ~0m00,157s  and    409633 steps for N=20 queens
                ~0m00,488s  and    812381 steps for N=26 queens
                ~0m04,062s  and   6142242 steps for N=28 queens
                ~1m26,536s  and 114841495 steps for N=30 queens
                ~0m22,126s  and 114841495 steps for N=30 queens and -O3
                ~0m27,730s  and 114841495 steps for N=30 queens and all optimizations listed below      
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
 * Prints a map of the queens until the M-th positioned queen
 */
void Map(int X[N], int Y[N], int M) {
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
                if (X[i]==x && Y[i]==y) {
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
bool isSafeFromPrevious(int posX[N], int posY[N], int x, int y) {
    int queensPositioned = x, thisQueen = queensPositioned;
    for (int q=0; q<queensPositioned; q++) {
        //printf("      Checking %d %d and %d %d \n",posX[q],posY[q],x,y);
        if (posX[q]==x || posY[q]==y) return false; //If row/column is endangered
        else if ( (x==posX[q]+(thisQueen-q) || x==posX[q]-(thisQueen-q)) && 
                    (y==posY[q]+(thisQueen-q) || y==posY[q]-(thisQueen-q))) return false; //If diagonal is endangered
    }
    return true;
}


/**
 * Searches for a safe spot in a specified column
 */
bool SolveColumn(int X[N], int Y[N], int current, int startingPoint) {
    int x = current;
    for (int y=startingPoint; y<N; y++) {
        if (isSafeFromPrevious(X, Y, x, y)) {
            X[current] = current;
            Y[current] = y;
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
    int posX[N] = {0}; //Each queen's X-position
    int posY[N] = {0}; //Each queen's Y-position
    int yMin[N] = {0}; //Used to block already visited states


    //printf("(1,1) is %s\n", isSafeFromPrevious(posX, posY, 1, 1) ? "safe" : "endangered");
    //printf("(3,1) is %s\n", isSafeFromPrevious(posX, posY, 3, 1) ? "safe" : "endangered");
    //printf("(2,0) is %s\n", isSafeFromPrevious(posX, posY, 2, 0) ? "safe" : "endangered");

    printf("Now solving the problem. Please wait...\n");
    int i = 1, steps = 0;
    while(i<N) {
        steps++;
        if (steps%10000==0) printf("\rStep: %3d, Queen: %2d", steps, i);
        //Map(posX, posY, N);
        //printf("i=%d  yMin=%d\n",i, yMin[i]);
        if (!SolveColumn(posX, posY, i, yMin[i])) {
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
    Map(posX, posY, N); 
    bool allSafe = true;
    for (int i=0; i<N-1; i++) {
        if (!isSafeFromPrevious(posX, posY, posX[i], posY[i])) {
            printf("WRONG SOLUTION!!\n");
            allSafe = false;
        }
    }
    if (allSafe) printf("The solution was validated and is accepted.\n");

}
