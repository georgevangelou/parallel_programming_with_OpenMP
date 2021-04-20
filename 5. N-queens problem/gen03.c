/*
    Description:
        This program implements my Genetic Algorithm method of solving the "N-Queens Problem"
        Abides by Lab 5 Exercise 3 requirements

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
        Compiles/Runs/Debugs with: gcc gen03.c -o gen03 -lm -fopt-info -pg && time ./gen03 && gprof ./gen03
        Inherits all features of previous version if not stated otherwise
        Removed debugging features after ensuring correct functionality of the previous version
        Correctly solves the problem
        The problem is solved uncomparably faster than the backtracking method
        Number of genes required to solve the problem fast depends on the number of the queens
        --> If (GENES>N and not GENES>>>1), the solution is found very fast

        Profiler output for N=30 queens, GENES=30 genes and no optimizations:
            Each sample counts as 0.01 seconds.
                %   cumulative      self               self    total           
             time      seconds   seconds    calls  us/call  us/call  name    
            71.50         0.05      0.05    16290     3.07     3.07  UtilityFunction
            28.60         0.07      0.02    16290     1.23     1.23  MutationFunction
             0.00         0.07      0.00   294120     0.00     0.00  RandomInteger
             0.00         0.07      0.00     8145     0.00     0.00  CrossoverFunction
             0.00         0.07      0.00      543     0.00    36.87  BreedGeneration
             0.00         0.07      0.00      543     0.00    92.17  CalculateAllUtilityValues
             0.00         0.07      0.00        1     0.00     0.00  GeneInitialization
             0.00         0.07      0.00        1     0.00     0.00  Map
        
        Profiler output for N=200 queens, GENES=600 genes and no optimizations:
            Each sample counts as 0.01 seconds.
                %   cumulative   self                  self    total           
            time      seconds   seconds     calls   s/call   s/call  name    
            63.57        31.10    31.10     608981     0.00     0.00  UtilityFunction
            30.54        46.05    14.94     627289     0.00     0.00  MutationFunction
            3.09        47.56     1.51  194284123     0.00     0.00  RandomInteger
            2.41        48.74     1.18        936     0.00     0.02  BreedGeneration
            0.45        48.96     0.22     305403     0.00     0.00  CrossoverFunction
            0.02        48.97     0.01                               frame_dummy
            0.00        48.97     0.00        993     0.00     0.03  CalculateAllUtilityValues
            0.00        48.97     0.00         11     0.00     0.00  GeneInitialization
            0.00        48.97     0.00          7     0.00     6.99  Solve

        Without any optimizations:
            For N=08 queens a solution is found after:
                ~0m00,005s and  15 generations, using  20 genes
            For N=16 queens a solution is found after:
                ~0m00,005s and  69 generations, using  20 genes
            For N=20 queens a solution is found after:
                ~0m00,006s and  43 generations, using  20 genes
            For N=26 queens a solution is found after:
                ~0m00,018s and 103 generations, using  30 genes
            For N=30 queens a solution is found after:
                ~0m00,030s and  98 generations, using  30 genes
            For N=35 queens a solution is found after:
                ~0m00,026s and 100 generations, using  40 genes
            For N=50 queens a solution is found after:
                ~0m00,325s and 636 generations, using  50 genes
            For N=100 queens a solution is found after:
                ~0m02,739s and 759 generations, using 100 genes
                ~0m00,891s and 119 generations, using 200 genes
                ~0m00,730s and  63 generations, using 300 genes
                ~0m00,686s and  44 generations, using 400 genes
                ~0m00,946s and  48 generations, using 500 genes 
            For N=200 queens a solution is found after:
                ~0m12,280s and 140 generations, using 600 genes 
            For N=300 queens a solution is found after:
                ~0m42,520s and x generations, using 900 genes    
*/	


// ****************************************************************************************************************    
//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
//#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
//#pragma GCC target("avx")  //Enable AVX


// ****************************************************************************************************************    
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "time.h"


// ****************************************************************************************************************    
#define N 50        //Number of queens
#define GENES 100    //Number of genes (must by even)


/**
 * Produces a random integer in the range [mini,maxi]
 */
int RandomInteger(int mini, int maxi) {
    int gap = maxi-mini;
    int randomInGap = (int) (gap * ((float)rand())/((float)RAND_MAX) ); //[0,gap]
    return mini + randomInGap; //[mini,mini+gap]==[mini,maxi]
}


/**
 * Initializes positional array given
 */
void GeneInitialization(int genes[GENES][N]) {
    for (int i=0; i<GENES; i++) {
        for (int j=0; j<N; j++) {
            genes[i][j] = RandomInteger(0,N-1);
        }
    }
}


/**
 * Prints a map of the queens until the M-th positioned queen
 */
void Map(int posY[N], int M, int geneId) {
    //for (int i=0; i<M; i++) printf("~~~~", i); 
    printf("\n========================\n"); 
    printf("------- GENE %d -------\n", geneId);
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
    for (int i=0; i<N; i++) printf("---"); printf("---\n");
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
 * Finds the number collisions between the queens
 */ 
int UtilityFunction(int posY[N]) {
    int collisions = 0;
    for (int crnt=1; crnt<N; crnt++) {
        for (int old=0; old<crnt; old++) {
            if (old==crnt || posY[old]==posY[crnt]) collisions++; //If row/column is endangered
            else if (posY[crnt]==posY[old]+(crnt-old) || posY[crnt]==posY[old]-(crnt-old)) collisions++; //If diagonal is endangered
        }
    }
    return collisions;
}


/**
 * Takes two parent genes and produces two child genes
 */ 
void CrossoverFunction(int gene1[N], int gene2[N]) {
    for (int i=1; i<N; i++) {
        if (abs(gene1[i-1]-gene1[i])<2 || abs(gene2[i-1]-gene2[i])<2) {
            int temp = gene1[i];
            gene1[i] = gene2[i];
            gene2[i] = temp;
        }
    }
}


/**
 * Takes a gene and mutates it
 */ 
void MutationFunction(int gene[N]) {
    // Mark all values missing from the gene, so they can be used to replace duplicates
    int inGene[N] = {0};

    // Un-mark all existing values
    for (int i=0; i<N; i++) {
        inGene[gene[i]] = 1;
    }
    
    // Find duplicates and replace them with non-used values
    for (int i=1; i<N; i++) {
        for (int j=0; j<i; j++) {
            if (gene[i]==gene[j]) {
                for (int k=0; k<N; k++){
                    if (inGene[k]==0) {
                        gene[i] = k;
                        inGene[k] = 1;
                        k = N;
                    }
                }
            }
        }
    }

    // Performs the actual swapping
    int barrier = RandomInteger(1,N-3); // [1, N-3]
    int swapA = RandomInteger(0,barrier);   // [0,barrier]
    int swapB = RandomInteger(barrier+1,N-1); // [barrier+1,N-1]
    int temp = gene[swapA];
    gene[swapA] = gene[swapB];
    gene[swapB] = temp;

}


/**
 * Breeds next generation
 */ 
void BreedGeneration(int genes[GENES][N], int utilityValues[GENES]) {
    int genesNew[GENES][N] = {-1};

    // For all pairs of genes to create
    for (int i=0; i<GENES-1; i+=2) {
        int index1 = -1, index2 = -1;
        float limit_value = INFINITY;
        float value1 = limit_value, value2 = limit_value;
        //...access all current genes and in a semi-stochastic way, pick two low-value parents
        for (int j=0; j<GENES; j++) {
            float value = (float) (10 + RandomInteger(10,20)*utilityValues[j] );
            if (value<=value1) {
                value2 = value1;
                index2 = index1;
                value1 = value;
                index1 = j;
            } else if (value<value2) {
                value2 = value;
                index2 = j;
            }
        }

        //...then copy the parents to the new array
        for (int k=0; k<N; k++) {
            genesNew[i][k]   = genes[index1][k];
            genesNew[i+1][k] = genes[index2][k];
        }
        
        //...breed and mutate their children
        CrossoverFunction(genesNew[i], genesNew[i+1]);
        MutationFunction(genesNew[i]);
        MutationFunction(genesNew[i+1]);
    }

    // Finally copy the new genes into the old ones
    for (int i=0; i<GENES; i++) {
        for (int j=0; j<N; j++) {
            genes[i][j] = genesNew[i][j];
        }
    }    
}


/**
 *  Calculate and store all current genes utility values
 */
int CalculateAllUtilityValues(int genes[GENES][N], int utilityValues[GENES]) {
    int bestUtilityValueFoundAt = 0;
    for (int i=0; i<GENES; i++) {
        utilityValues[i] = UtilityFunction(genes[i]);
        if (utilityValues[i] < utilityValues[bestUtilityValueFoundAt]) {
            bestUtilityValueFoundAt = i;
        }
    }
    return bestUtilityValueFoundAt;
}


/**
 * The main program
 */
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements my Genetic Algorithm method of solving the \"N-Queens Problem\".\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    __time_t now = time(0);
    srand(now);
    int genes[GENES][N];
    int utilityValues[GENES] = {1};
    int generation = 0, bestGene = 0;

    //Create a random set of genes    
    GeneInitialization(genes);

    printf("Queens set at: %d   Genes set at: %d\n", N, GENES);
    printf("Now solving the problem. Please wait...\n");
    
    //While no solution is found
    while(utilityValues[bestGene]!=0) {

        //...for each repetition create the next generation of genes
        BreedGeneration(genes, utilityValues);

        //...and calculate all genes's utility values
        bestGene = CalculateAllUtilityValues(genes, utilityValues);
        
        if ((++generation)%100==0) printf("\r> Current Generation: %3d  Best utility value: %2d", generation, utilityValues[bestGene]);
    }
    
    printf("\r> Current Generation: %3d  Best utility value: %2d\nSOLUTION FOUND. ", generation, utilityValues[bestGene]);
    printf("The solution found is:\n"); 
    Map(genes[bestGene], N, bestGene); 
    printf("\nMap printed. Program finished"); 
    
    return 0;
}