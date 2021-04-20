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
        Compiles/Runs/Debugs with: gcc nn02.c -o nn02 -lm -pg && time ./nn02 && gprof ./nn02
        Inherits all settings of previous version if not stated otherwise
        Executes the activation and training (forward- and back- propagation) of the Neural Network
        Produces correct results when the unity neuron is omitted
*/	


#include "stdio.h"
#include "stdlib.h"
#include "math.h"


#define INPUT_SIZE 2        //Number of inputs
#define LAYER1_NEURONS 2    //Number of 1st layer neurons
#define LAYER2_NEURONS 5    //Number of 2nd layer neurons
#define DEBUG 0            //Debugging options
#define LEARNING_RATE 0.5   //Learning rate
#define UNITY_NEURON 0      //The unity neuron
#define MAX_STEPS 5


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
 * TODO: CHECK FOR CORRECT INDEXING WITH THE UNITY NEURON
 **/
void CalculateInnerStates(double *Inputs, double *InnerStates, double *Weights, int InputSize, int neurons) {
    if (DEBUG==4) printf("Now calculating a layer's outer states...\n");
    for (int i=0; i<neurons; i++) {
        InnerStates[i] = UNITY_NEURON ? Weights[i*InputSize + InputSize-1] : 0.0; //The "unity" neuron
        for (int j=0; j<InputSize-UNITY_NEURON; j++) {
            InnerStates[i] += Inputs[j] * Weights[i*(InputSize) + j];
        }
    }
}


/**
 * Calculates the outer state of all neurons <i> of a given layer
 **/
void CalculateOuterStates(double *InnerStates, double *OuterStates, int neurons) {
    if (DEBUG==4) printf("Now calculating a layer's outer states...\n");
    for (int i=0; i<neurons; i++) {
        OuterStates[i] = 1 / (1+exp(-InnerStates[i]));
    }
}


/**
 * Activates a specific layer
 **/
void ActivateLayer(double *Inputs, double *InnerStates, double *Outputs, double *Weights, int inputSize, int neurons) {
    if (DEBUG==3) printf("Now calculating a layer's output...\n");
    CalculateInnerStates(Inputs, InnerStates, Weights, inputSize, neurons);
    CalculateOuterStates(InnerStates, Outputs, neurons);
}


/**
 * Activates the whole Neural Network
 **/
void ActivateNeuralNetwork(double *Inputs) {
    if (DEBUG==2) printf("Now calculating the network's output...\n");
    ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], INPUT_SIZE+UNITY_NEURON, LAYER1_NEURONS);
    ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS+UNITY_NEURON, LAYER2_NEURONS);
}


/**
 * Initializes the weights of single layer
 **/
void InitializeLayerWeights(double *Weights, int SizeA, int SizeB) {
    if (DEBUG==3) printf("Now initializing the weights of a specific layer...\n");
    for (int i=0; i<SizeA; i++) {
        for (int j=0; j<SizeB; j++) {
            Weights[i*(SizeB) + j] = ((double)rand()) / ((double)RAND_MAX);
        }
    }
}


/**
 * Initializes the weights of the Neural Network
 **/
void InitializeAllWeights() {
    if (DEBUG==2) printf("Now initializing the weights of all layers...\n");
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
void InnerLayerErrors(double *curOutputs, double *nextWeights, double *curErrors, double *nextErrors, int curNeuronsNum, int nextNeuronsNum) {
    for (int c=0; c<curNeuronsNum; c++) {
        double myError = 0.0;
        for (int n=0; n<nextNeuronsNum; n++) {
            myError += nextWeights[n*(curNeuronsNum+UNITY_NEURON) + c] * nextErrors[n];
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
    UpdateLayer(&WL1[0][0], &EL1[0], inputs, LAYER1_NEURONS+UNITY_NEURON, INPUT_SIZE+UNITY_NEURON);
    UpdateLayer(&WL2[0][0], &EL2[0], &OL1[0], LAYER2_NEURONS+UNITY_NEURON, LAYER1_NEURONS+UNITY_NEURON);
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
 * The main program
 **/
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements a Neural Network of %d Layers.\n", 2);
	printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", INPUT_SIZE, LAYER1_NEURONS, LAYER2_NEURONS);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    __time_t seconds;
    srand(seconds);

    double *Inputs = (double*) calloc(INPUT_SIZE, sizeof(double)); 
    double *GoldenOutputs = (double*) calloc(LAYER2_NEURONS, sizeof(double)); 

    int steps = 0;
    int max_steps = MAX_STEPS;

    InitializeAllWeights();
    AcquireInputs(Inputs, INPUT_SIZE);
    AcquireGoldenOutputs(GoldenOutputs, LAYER2_NEURONS);
    
    printf("Set number of steps: ");
    scanf("%d", &max_steps);
    


    printf("\nThe input was:\n");
    for (int i=0; i<INPUT_SIZE; i++) 
        printf("%.10lf\n", Inputs[i]);

    do {
        printf(">>>> STEP: %7d  <<<<\n", steps);
        ActivateNeuralNetwork(Inputs);
        TrainNeuralNetwork(Inputs, GoldenOutputs);
        
        if (DEBUG==6) {
            printf("\nThe weights of the hidden layer are:\n");
            for (int i=0; i<LAYER1_NEURONS; i++) 
                for (int j=0; j<INPUT_SIZE+UNITY_NEURON; j++)
                    printf("%.10lf\n", WL1[i][j]);

            printf("\nThe weights of the output layer are:\n");
            for (int i=0; i<LAYER2_NEURONS; i++) 
                for (int j=0; j<LAYER1_NEURONS+UNITY_NEURON; j++)
                    printf("%.10lf\n", WL2[i][j]);

            printf("\nThe output is:\n");
            for (int i=0; i<LAYER2_NEURONS; i++) 
                printf("%.10lf\n", OL2[i]);
            printf("\n");
        }

    } while (steps++<max_steps);

    printf("\nThe final output compared to the golden output is:\n");
    for (int i=0; i<LAYER2_NEURONS; i++) 
        printf("Golden: %.10lf  <>  Real: %.10lf  \n", GoldenOutputs[i], OL2[i]);
    printf("\n");

    free(Inputs);
    free(GoldenOutputs);
    return 0;
}



