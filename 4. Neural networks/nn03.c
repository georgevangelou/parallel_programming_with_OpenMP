/*
    Description:
        This program implements a Neural Network of one hidden and one output layer
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
        Compiles/Runs/Debugs with: gcc nn03.c -o nn03 -lm -O3 -fopt-info -pg && time ./nn03 && gprof ./nn03
        Inherits all settings of previous version if not stated otherwise
        Executes the activation and training (forward- and back- propagation) of the Neural Network
        Produces correct results with or without the unity neuron
        The vast majority of the execution time is consumed by: UpdateLayer() and CalculateInnerStates()
*/	


// ****************************************************************************************************************    
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
#pragma GCC target("avx")  //Enable AVX


// ****************************************************************************************************************    
#include "stdio.h"
#include "stdlib.h"
#include "math.h"


// ****************************************************************************************************************    
#define INPUT_SIZE 800        //Number of inputs
#define LAYER1_NEURONS 100    //Number of 1st layer neurons
#define LAYER2_NEURONS 10    //Number of 2nd layer neurons
#define DEBUG 0            //Debugging options
#define LEARNING_RATE 0.1   //Learning rate
#define UNITY_NEURON 1      //The unity neuron


double WL1[LAYER1_NEURONS][INPUT_SIZE+UNITY_NEURON];       //The weights of the 1st layer
double WL2[LAYER2_NEURONS][LAYER1_NEURONS+UNITY_NEURON];   //The weights of the 2nd layer

double DL1[LAYER1_NEURONS]; //The inner states of 1st layer
double DL2[LAYER2_NEURONS]; //The inner states of 2nd layer

double OL1[LAYER1_NEURONS]; //The outer states of 1st layer
double OL2[LAYER2_NEURONS]; //The outer states of 2nd layer

double EL1[LAYER1_NEURONS]; //The errors of the 1st layer
double EL2[LAYER2_NEURONS]; //The errors of the 2nd layer



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
    ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], INPUT_SIZE+UNITY_NEURON, LAYER1_NEURONS);
    ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS+UNITY_NEURON, LAYER2_NEURONS);
}


/**
 * Initializes the weights of single layer
 **/
void InitializeLayerWeights(double *Weights, int neurons, int inps) {
    for (int i=0; i<neurons; i++) {
        for (int j=0; j<inps; j++) {
            Weights[i*(inps) + j] = ((double)rand()) / ((double)RAND_MAX);
        }
    }
}


/**
 * Initializes the weights of the Neural Network
 **/
void InitializeAllWeights() {
    InitializeLayerWeights(&WL1[0][0], LAYER1_NEURONS, INPUT_SIZE+UNITY_NEURON);
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
void UpdateLayer(double *weights, double *errors, double *inputs, int neurons, int inputsNum) {
    for (int i=0; i<neurons; i++) {
        for (int j=0; j<inputsNum; j++) {
            weights[i*inputsNum + j] += LEARNING_RATE * errors[i] * inputs[j];
        }
    }
}


/**
 * Updates all layers's weights
 **/ 
void UpdateLayers(double *inputs) {
    UpdateLayer(&WL1[0][0], &EL1[0], inputs, LAYER1_NEURONS, INPUT_SIZE+UNITY_NEURON);
    UpdateLayer(&WL2[0][0], &EL2[0], &OL1[0], LAYER2_NEURONS, LAYER1_NEURONS+UNITY_NEURON);
}


/**
 * Performs the Back-Propagation algorithm to calculate the errors
 **/
void ErrorBackPropagation(double *GoldenOutputs) {
    OutputLayerErrors(&OL2[0], GoldenOutputs, &EL2[0], LAYER2_NEURONS);
    InnerLayerErrors(&OL1[0], &WL2[0][0], &EL1[0], &EL2[0], LAYER1_NEURONS, LAYER2_NEURONS);
}


/**
 * Trains the Neural Network by executing the back-propagation algorithm
 * and re-calculating all weights
 **/ 
void TrainNeuralNetwork(double *inputs, double *GoldenOutputs) {
    ErrorBackPropagation(GoldenOutputs);
    UpdateLayers(inputs);
}


/**
 * Acquires the inputs
 **/
void AcquireInputs(double *array, int size) {
    for (int i=0; i<size; i++)
        array[i] = -1 + 2 * ((double) rand()) / ((double) RAND_MAX);
}


/**
 * Acquired the correct outputs
 **/
void AcquireGoldenOutputs(double *array, int size) {
    for (int i=0; i<size; i++)
        array[i] = ((double) rand()) / ((double) RAND_MAX);
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
 * The main program
 **/
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements a Neural Network of %d Layers.\n", 2);
	printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", INPUT_SIZE, LAYER1_NEURONS, LAYER2_NEURONS);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    __time_t seconds;
    srand(seconds);

    double *DataIn = (double*) calloc(INPUT_SIZE, sizeof(double)); 
    double *GoldenData = (double*) calloc(LAYER2_NEURONS, sizeof(double)); 

    int steps = 0;
    int max_steps = 0;

    InitializeAllWeights();
    AcquireInputs(DataIn, INPUT_SIZE);
    AcquireGoldenOutputs(GoldenData, LAYER2_NEURONS);
    
    printf("Set number of steps: ");
    scanf("%d", &max_steps);
    
    if (DEBUG==6) {
        printf("\nThe input was:\n");
        for (int i=0; i<INPUT_SIZE; i++) 
            printf("%.10lf\n", DataIn[i]);
        printf("\n");
    }


    ActivateNeuralNetwork(DataIn); //First activation
    do {
        steps++;
        TrainNeuralNetwork(DataIn, GoldenData);
        ActivateNeuralNetwork(DataIn);
        
        /*
        if (steps%1000==0) {
            printf("STEP %8d ==> ", steps);
            printf("Mean Square Error: %.20lf\n", MeanSquareError(OL2, GoldenData, LAYER2_NEURONS));
        }*/

    } while (steps<max_steps);

    printf("\nSteps completed: %d", steps);
    printf("\nThe final output compared to the golden output is:\n");
    for (int i=0; i<LAYER2_NEURONS; i++) 
        printf("  Golden: %.13lf  <o>  Real: %.13lf  \n", GoldenData[i], OL2[i]);
    printf("\n");
    printf("The final Mean Square Error is: %.15lf\n", MeanSquareError(OL2, GoldenData, LAYER2_NEURONS));

    free(DataIn);
    free(GoldenData);
    return 0;
}



