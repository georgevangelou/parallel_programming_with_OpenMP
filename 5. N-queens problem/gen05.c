/*
    Description:
        This program implements my Genetic Algorithm method of solving the "N-Queens Problem"
         

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
        Compiles/Runs/Debugs with: gcc gen05.c -o gen05 -lm -fopt-info -fopenmp -pg && time ./gen05 && gprof ./gen05
        Inherits all features of previous version if not stated otherwise
        Almost identical to gen04, but achieves double execution speed after creating and using a 
            custom thread-safe random-generator function

        Without any optimizations and 12 threads reported:
            For N=200 queens a solution is found after:
                ~0m13,301s and  89 generations, using 600 genes per thread(1054 summed generations)
        
        CHECK gen04 FOR MORE DETAILS... (only difference is the custom thread-safe random-generator function and, thus,
            the faster execution speed)
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
#include "omp.h"


// ****************************************************************************************************************    
#define N 16        //Number of queens
#define GENES 20    //Number of genes (must by even)
#define TARGET_THREADS 12   //Number of threads to ask


/**
 * Produces a random integer in the range [mini,maxi]
 */
int RandomInteger2(int mini, int maxi, unsigned *seed) {
    *seed = (unsigned) (1664525 * (*seed) + 1013904223 )% RAND_MAX;
    int gap = maxi-mini;
    int randomInGap = (int) (gap * ((float)(*seed))/((float)RAND_MAX) ); //[0,gap]
    return mini + randomInGap; //[mini,mini+gap]==[mini,maxi]
}


/**
 * Initializes positional array given
 */
void GeneInitialization(int genes[GENES][N]) {
    unsigned seed = 1046900;
    for (int i=0; i<GENES; i++) {
        for (int j=0; j<N; j++) {
            genes[i][j] = RandomInteger2(0,N-1, &seed);
        }
    }
}


/**
 * Prints a map of the queens until the M-th positioned queen
 */
void Map3(int posY[N], int M) {
    for (int i=0; i<N; i++) printf("==="); printf("===\n---");
    for (int i=0; i<N/3; i++) printf("---"); printf("  FITTEST GENE  ");
    for (int i=0; i<N/3; i++) printf("---"); printf("---\n===");
    for (int i=0; i<N; i++) printf("==="); printf("\n");
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
void MutationFunction(int gene[N], unsigned *seed) {
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
    int barrier = RandomInteger2(1,N-3, seed); // [1, N-3]
    int swapA = RandomInteger2(0,barrier, seed);   // [0,barrier]
    int swapB = RandomInteger2(barrier+1,N-1, seed); // [barrier+1,N-1]
    int temp = gene[swapA];
    gene[swapA] = gene[swapB];
    gene[swapB] = temp;

}


/**
 * Breeds next generation
 */ 
void BreedGeneration(int genes[GENES][N], int utilityValues[GENES], unsigned *seed) {
    int genesNew[GENES][N] = {-1};

    // For all pairs of genes to create
    for (int i=0; i<GENES-1; i+=2) {
        int index1 = -1, index2 = -1;
        float limit_value = INFINITY;
        float value1 = limit_value, value2 = limit_value;
        //...access all current genes and in a semi-stochastic way, pick two low-value parents
        for (int j=0; j<GENES; j++) {
            float value = (float) (10 + RandomInteger2(10,20, seed)*utilityValues[j] );
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
        MutationFunction(genesNew[i], seed);
        MutationFunction(genesNew[i+1], seed);
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
unsigned CalculateAllUtilityValues(int genes[GENES][N], int utilityValues[GENES]) {
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
 * Runs the genetic algorithm to solve the problem
 */
long int Solve(int fittestGene[N], unsigned threadID, int *whoHasFinished, unsigned *solversGenerations) {
    unsigned seed = threadID;
    int genes[GENES][N];
    int utilityValues[GENES] = {1};

    //Create a random set of genes    
    GeneInitialization(genes);

    long int generation = 0;
    unsigned bestGene = 0;

    //While no solution is found
    while(utilityValues[bestGene]!=0 && *whoHasFinished<0) {
        generation++; 

        //...for each repetition create the next generation of genes
        BreedGeneration(genes, utilityValues, &seed);

        //...and calculate all genes's utility values
        bestGene = CalculateAllUtilityValues(genes, utilityValues);
    }

    //After a correct gene has been found, store its details in variables visible to main()
    #pragma omp critical
    {
        *whoHasFinished = threadID;
        *solversGenerations = generation;
        for (int i=0; i<N; i++) fittestGene[i] = genes[bestGene][i];
    }

    return generation;
}


/**
 * The main program
 */
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements my Genetic Algorithm method of solving the \"N-Queens Problem\".\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    int fittestGene[N] = {0};
    int numberOfThreads = 1, whoHasFinished = -1;
    unsigned solversGenerations = 0;
    long int totalGenerations = 0;

    printf("Queens set at: %d   Genes set at: %d\n", N, GENES);
    printf("Now solving the problem. Please wait...\n");
    
    #pragma omp parallel num_threads(TARGET_THREADS) reduction(+:totalGenerations)
    {
        //Check how many threads were created
        #pragma omp single
        numberOfThreads = omp_get_num_threads(); 

        //Tell each thread to start searching for a solution
        totalGenerations = Solve(fittestGene, omp_get_thread_num(), &whoHasFinished, &solversGenerations);
    }
    
    printf("Algorithm completed. Number of threads used: %d   Total generations: %ld\n", numberOfThreads, totalGenerations);
    printf("Solution found by thread #%d in #%u generations.\n", whoHasFinished, solversGenerations);
    printf("The solution found is:\n");
    //Map3(fittestGene, N); 
    
    return 0;
}

