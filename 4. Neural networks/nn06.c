/*
    Description:
        This program implements a Neural Network of one hidden and one output layer

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
        Compiles/Runs/Debugs with: gcc nn06.c -o nn06 -lm -O3 -fopenmp -fopt-info -pg && time ./nn06 && gprof ./nn06
        Inherits all settings of previous version if not stated otherwise
        Performs the training of the network in batches
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
#include "string.h"


// ****************************************************************************************************************    
#define LAYER1_NEURONS 100    //Number of 1st layer neurons
#define LAYER2_NEURONS 10    //Number of 2nd layer neurons
#define DEBUG 0            //Debugging options
#define LEARNING_RATE 0.3   //Learning rate (0.0001)
#define UNITY_NEURON 1      //The unity neuron

#define TRAIN_FILE_PATH "fashion-mnist_train.csv"
#define TEST_FILE_PATH  "fashion-mnist_test.csv"
#define TRAIN_DATA_NUMBER 60000
#define TEST_DATA_NUMBER  10000

#define PIXELS 784
#define EPOCHS 10
#define BATCH_SIZE 10000

double TRAIN_DATA[TRAIN_DATA_NUMBER][PIXELS+1]; //The train images (class and pixels)
double TEST_DATA[TEST_DATA_NUMBER][PIXELS+1];   //The test images (class and pixels)

double TRAIN_GOLDEN_OUTPUTS[TRAIN_DATA_NUMBER][LAYER2_NEURONS];
double TEST_GOLDEN_OUTPUTS[TEST_DATA_NUMBER][LAYER2_NEURONS];

double WL1[LAYER1_NEURONS][PIXELS+UNITY_NEURON];       //The weights of the 1st layer
double WL2[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   //The weights of the 2nd layer

double WL1delta[LAYER1_NEURONS][PIXELS+UNITY_NEURON];       //The new weights of the 1st layer
double WL2delta[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   //The new weights of the 2nd layer

double DL1[LAYER1_NEURONS]; //The inner states of 1st layer
double DL2[LAYER2_NEURONS]; //The inner states of 2nd layer

double OL1[LAYER1_NEURONS]; //The outer states of 1st layer
double OL2[LAYER2_NEURONS]; //The outer states of 2nd layer

double EL1[LAYER1_NEURONS]; //The errors of the 1st layer
double EL2[LAYER2_NEURONS]; //The errors of the 2nd layer


//TODO: DELETE THIS
/**
 * Zeros all stored neuron errors
 **/
void InitializeAllErrors() {
    for (int i=0; i<LAYER1_NEURONS; i++) EL1[i] = 0;
    for (int i=0; i<LAYER2_NEURONS; i++) EL2[i] = 0;
}


/**
 * Zeros all stored weights deltas
 **/
void InitializeAllDeltas() {
    for (int i=0; i<LAYER1_NEURONS; i++) 
        for (int j=0; j<PIXELS+UNITY_NEURON; j++)
            WL1delta[i][j] = 0;
    for (int i=0; i<LAYER2_NEURONS; i++) 
        for (int j=0; j<LAYER1_NEURONS+UNITY_NEURON; j++)
            WL2delta[i][j] = 0;
}



/**
 * Calculates the derivative of a neuron output
 **/
double NeuronOutputDerivative(double output) {
    return output * (1.0 - output);
}


/**
 * Calculates the inner state of all neurons <i> of a given layer
 **/
void CalculateInnerStates(double *Inputs, double *InnerStates, double *Weights, int InputSize, int neurons) {
    #pragma omp parallel for schedule(static, 10)
    for (int i=0; i<neurons; i++) {
        InnerStates[i] = 1.0 * UNITY_NEURON ? Weights[i*InputSize + InputSize - UNITY_NEURON] : 0.0; //The "unity" neuron
        for (int j=0; j<InputSize-UNITY_NEURON; j++) {
            InnerStates[i] += Inputs[j] * Weights[i*(InputSize) + j];
        }
    }
}


/**
 * Calculates the outer state of all neurons <i> of a given layer
 **/
void CalculateOuterStates(double *InnerStates, double *OuterStates, int neurons) {
    for (int i=0; i<neurons; i++) {
        OuterStates[i] = 1 / (1+exp(-InnerStates[i]));
    }
}


/**
 * Activates a specific layer
 **/
void ActivateLayer(double *Inputs, double *InnerStates, double *Outputs, double *Weights, int inputSize, int neurons) {
    CalculateInnerStates(Inputs, InnerStates, Weights, inputSize, neurons);
    CalculateOuterStates(InnerStates, Outputs, neurons);
}


/**
 * Activates the whole Neural Network
 **/
void ActivateNeuralNetwork(double *Inputs) {
    ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], PIXELS+UNITY_NEURON, LAYER1_NEURONS);
    ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS+UNITY_NEURON, LAYER2_NEURONS);
}


/**
 * Initializes the weights of single layer
 **/
void InitializeLayerWeights(double *Weights, int neurons, int inps) {
    for (int i=0; i<neurons; i++) {
        for (int j=0; j<inps; j++) {
            Weights[i*(inps) + j] = -1 + 2 * ((double)rand()) / ((double)RAND_MAX);
        }
    }
}


/**
 * Initializes the weights of the Neural Network
 **/
void InitializeAllWeights() {
    InitializeLayerWeights(&WL1[0][0], LAYER1_NEURONS, PIXELS+UNITY_NEURON);
    InitializeLayerWeights(&WL2[0][0], LAYER2_NEURONS, LAYER1_NEURONS+UNITY_NEURON);
}


/**
 * Calculates the output layer's errors
 **/
void OutputLayerErrors(double *outputs, double *expected, double *errors, int neurons) {
    for (int i=0; i<neurons; i++) {
        errors[i] = (expected[i] - outputs[i]) * NeuronOutputDerivative(outputs[i]);
    }
}


/**
 * Calculates a hidden layer's errors
 **/
void InnerLayerErrors(double *curOutputs, double *nextWeights, double *curErrors, double *nextErrors, int curNeurons, int nextNeurons) {
    for (int c=0; c<curNeurons; c++) {
        double myError = 0.0;
        for (int n=0; n<nextNeurons; n++) {
            myError += nextWeights[n*curNeurons + c] * nextErrors[n];
        }
        curErrors[c] = myError * NeuronOutputDerivative(curOutputs[c]);
    }
}


/**
 * Updates a layer's weights
 **/ 
void UpdateLayer2(double *weightsDelta, double *errors, double *inputs, int neurons, int inputsNum) {
    #pragma omp parallel for schedule(static, 10)
    for (int i=0; i<neurons; i++) {
        for (int j=0; j<inputsNum; j++) {
            weightsDelta[i*inputsNum + j] += LEARNING_RATE * errors[i] * inputs[j];
        }
    }
}


/**
 * Uses accumulated weights deltas to update all weights 
 **/
void UseDeltas() {
    if (DEBUG==1) printf("Using deltas to update weights...\n");
    for (int i=0; i<LAYER1_NEURONS; i++) {
        for (int j=0; j<PIXELS; j++) {
           WL1[i][j] += WL1delta[i][j] / BATCH_SIZE;
        }
    }
    for (int i=0; i<LAYER2_NEURONS; i++) {
        for (int j=0; j<LAYER1_NEURONS; j++) {
           WL2[i][j] += WL2delta[i][j] / BATCH_SIZE;
        }
    }
}


/**
 * Updates all layers's weights
 **/ 
void UpdateLayers2(double *inputs) {
    if (DEBUG==1) printf("Accumulating weights deltas...\n");
    UpdateLayer2(&WL1delta[0][0], &EL1[0], inputs, LAYER1_NEURONS, PIXELS+UNITY_NEURON);
    UpdateLayer2(&WL2delta[0][0], &EL2[0], &OL1[0], LAYER2_NEURONS, LAYER1_NEURONS+UNITY_NEURON);
}


/**
 * Performs the Back-Propagation algorithm to calculate the errors
 **/
void ErrorBackPropagation(double *GoldenOutputs) {
    if (DEBUG==1) printf("Running Error back-propagation...\n");
    OutputLayerErrors(&OL2[0], GoldenOutputs, &EL2[0], LAYER2_NEURONS);
    InnerLayerErrors(&OL1[0], &WL2[0][0], &EL1[0], &EL2[0], LAYER1_NEURONS, LAYER2_NEURONS);
}


/**
 * Trains the Neural Network by executing the back-propagation algorithm
 * and re-calculating all weights
 **/ 
void TrainNeuralNetwork2(double *inputs, double *GoldenOutputs) {
    ErrorBackPropagation(GoldenOutputs);
    UpdateLayers2(inputs);
}


/**
 * Acquires the train data
 **/
void AcquireTrainData() {
    FILE *fp = fopen(TRAIN_FILE_PATH, "r");
    char *token;
    if(fp != NULL) {
        char line[PIXELS*6];
        int picture = 0;
        while(fgets(line, sizeof line, fp) != NULL) {
            token = strtok(line, ",");
            int element = 0;
            while(token != NULL) {
                TRAIN_DATA[picture][element++] = atoi(token);
                token = strtok(NULL, ",");
            }
            picture++;  
        }
        fclose(fp);
    }
}


/**
 * Acquires the test data
 **/
void AcquireTestData() {
    FILE *fp = fopen(TEST_FILE_PATH, "r");
    char *token;
    if(fp != NULL) {
        char line[PIXELS*6];
        int picture = 0;
        while(fgets(line, sizeof line, fp) != NULL) {
            token = strtok(line, ",");
            int element = 0;
            while(token != NULL) {
                TEST_DATA[picture][element++] = atoi(token);
                token = strtok(NULL, ",");
            }
            picture++;
        }
        fclose(fp);
    }
}


/**
 * Acquired the correct train outputs
 **/
void AcquireTrainGoldenOutputs() {
    for (int p=0; p<TRAIN_DATA_NUMBER; p++) {
        for (int i=0; i<LAYER2_NEURONS; i++) {
            if (i == (int) TRAIN_DATA[p][0]) {
                TRAIN_GOLDEN_OUTPUTS[p][i] = 0.9;
            }
            else 
                TRAIN_GOLDEN_OUTPUTS[p][i] = 0.1;
        }
    }
}


/**
 * Acquired the correct test outputs
 **/
void AcquireTestGoldenOutputs() {
    for (int p=0; p<TEST_DATA_NUMBER; p++)
        for (int i=0; i<LAYER2_NEURONS; i++) {
            if (i == (int) TEST_DATA[p][0]) 
                TEST_GOLDEN_OUTPUTS[p][i] = 0.999;
            else 
                TEST_GOLDEN_OUTPUTS[p][i] = 0.001;
        }
}


/**
 * Mean Square Error
 **/
double MeanSquareError(double *RealOutputs, double *GoldenOutputs, int outputSize) {
    double error = 0.0;
    for (int i=0; i<outputSize; i++)
        error += (RealOutputs[i]-GoldenOutputs[i]) * (RealOutputs[i]-GoldenOutputs[i]);
    return sqrt(error);
}


/**
 * Finds index of max in array
 **/
int arrayMax(double *array, int size) {
    double mx = -INFINITY;
    int mx_index = -1;
    for (int i=0; i<size; i++) {
        if (array[i] > mx) {
            mx = array[i];
            mx_index = i;
        }
    }
    return mx_index;
}


/**
 * The main program
 **/
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements a Neural Network of %d Layers.\n", 2);
	printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", PIXELS, LAYER1_NEURONS, LAYER2_NEURONS);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    __time_t seconds;
    srand(seconds);

    int step  = 0;
    int epoch = 1;
    int hits = 0;
    

    InitializeAllWeights();
    InitializeAllDeltas();
    AcquireTrainData();
    AcquireTestData();
    AcquireTrainGoldenOutputs();
    AcquireTestGoldenOutputs();
    
    printf("~~~~~ NOW TRAINING THE NETWORK... ~~~~~\n");
    do {
        InitializeAllErrors();
        //int picture = step % TRAIN_DATA_NUMBER;
        int picture = rand() % TRAIN_DATA_NUMBER;
        double *DataIn = &TRAIN_DATA[picture][1];
        double *GoldenData = &TRAIN_GOLDEN_OUTPUTS[picture][0]; 
        ActivateNeuralNetwork(DataIn);
        TrainNeuralNetwork2(DataIn, GoldenData);
        step ++;
        if ((int) TRAIN_DATA[picture][0] == arrayMax(OL2, LAYER2_NEURONS)) hits++;
        //printf("Real class: %.0lf, Predicted class: %d\n", TRAIN_DATA[picture][0], arrayMax(OL2, LAYER2_NEURONS));
        
        if (step % BATCH_SIZE == 0) {
            UseDeltas();                
            InitializeAllDeltas();
        }

        if (step == TRAIN_DATA_NUMBER) {
            printf("TRAINING EPOCH: %3d,  STEP: %6d", epoch, step);
            printf(" ==> ACCURACY: %.2lf%%\n", 100*hits/((double) step)); 
            epoch++;
            hits = 0;
            step = 0;
        }
    } while (epoch<=EPOCHS);


    printf("\n~~~~~ NOW EVALUATING THE NETWORK... ~~~~~\n");
    step = 0;
    hits = 0;
    do {
        int picture = step;
        double *DataIn = &TEST_DATA[picture][1];
        double *GoldenData = &TEST_GOLDEN_OUTPUTS[picture][0]; 
        ActivateNeuralNetwork(DataIn);
        step ++;
        if ((int) TEST_DATA[picture][0] == arrayMax(OL2, LAYER2_NEURONS)) hits++;
    } while (step<TEST_DATA_NUMBER);
    printf("EVALUATION STEP: %6d", step);
    printf(" ==> ACCURACY: %.2lf%%\n", 100*hits/((double) step));   
    
    return 0;
}



