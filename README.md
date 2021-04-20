# Parallel Programming with OpenMP

This repository lists 4 problems solved using C. Each problem has its own serial and parallel implementations. For the latter, the [OpenMP](https://www.openmp.org/) API was utilized.
The source code of this repository was developed for the "Parallel Programming for Machine Learning Problems" course conducted at the Department of Electrical and Computer Engineering, University of Patras, Greece.

## Contents:
### 1. **[K-means](https://en.wikipedia.org/wiki/K-means_clustering) serial implementation**
##### The algorithm
- **Step 0:** Create N random vectors: Vec[N][Nv].
- **Step 1:** Initialiaze the centers. Choose Nc unique vectors from Vec: Center[Nc][Nv].
- **Step 2:** For each Vec[N][Nv], calculate the minimum Euclidean distance from Center[Nc][Nv] and update Classes[N] based on the minimum distance.
- **Step 3:** Update Center[Nc][Nv] by calculating the average of the vectors that belong to the same class.
- **Step 4:** If the sum of the vector distances from their corresponding centers is less than a certain threshold, the algorithm finishes. If not go to step 2.
##### Versions: _See 'Version Notes' inside each file for details_
### 2. **[K-means](https://en.wikipedia.org/wiki/K-means_clustering) with OpenMP:** A parallel implementation of the K-Means clustering algorithm
##### Versions: _See 'Version Notes' inside each file for details_
### 3. **[Travelling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)**
##### Algorithm #1: Random search
- **Step 0:** Define a random route.
- **Step 1:** Calculate the total distance: tot_dist.
- **Step 2:** Swap two random cities.
- **Step 3:** Calculate the new total distance: tot_dist_new.
- **Step 4:** If tot_dist_new > tot_dist: undo the swap.
- **Step 5:** Repeat Step 2 to 4 for a fixed number of times.
##### Algorithm #2a: Heinritz-Hsiao Algorithm
- **Step 0:** Start from the initial city.
- **Step 1:** Find the nearest not-visited city. Travel to this city and update the current total travelled distance.
- **Step 2:** If you have visited every city add the distance from the city you are currently at to the initial city, to the total travelled distance. In this case, the algorithm finishes, otherwise go to Step 1.
##### Algorithm #2b
This algorithm is the same as Algorithm #2a except for:
- **Step 1:** Find the two nearest cities. Choose whether to travel to the nearest or the second nearest city with a probability other than 50%. Travel to the city of choice and
##### Versions: _See 'Version Notes' inside each file for details_
### 4. **[Neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network)** 
##### Neural network structure
The neural network is a fully-connected network consisting of 2 layers with 100 and 10 neurons, respectively. The input vector holds 12 values. The weights are stored in WL1[100][12+1] (+1 is for the bias) and WL2[10][100+1] for layer 1 and 2, respectively. The internal states of the neurons are stored in DL1[100] for layer 1 and DL2[10] for layer 2, whereas their corresponding outputs in OL1[100] and OL2[10].
##### Versions: _See 'Version Notes' inside each file for details_
### 5. **[N queens problem](https://en.wikipedia.org/wiki/Eight_queens_puzzle)** 
##### The algorithm
- **Step 0:** Place a queen to (1,1), first row and first column, that is.
- **Step 1:** Place the next queen to the next column without being neither diagonally nor on the same row with another queen.
- **Step 2:** If Step 1 is not possible, move the queen of the previous column to a new row without violating the "not diagonal, not on tha same row" constraints. Then, proceed to Step 1.
- **Step 3:** The algorithm finishes either when a queen has been placed in every column, or when there is no position on the first column for the first queen. In the second case, there is no solution for the specific NxN chessboard.
##### Versions: _See 'Version Notes' inside each file for details_
