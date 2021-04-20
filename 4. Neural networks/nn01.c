/*
    Description:
        This program implements a Neural Network of one hidden and one output layer
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
        Compiles/Runs/Debugs with: gcc nn01.c -o nn01 -lm -pg && time ./nn01 && gprof ./nn01
        Executes the activation of the Neural Network and produces correct results
*/	


#include "stdio.h"
#include "stdlib.h"
#include "math.h"


#define INPUT_SIZE 2        //Number of inputs
#define LAYER1_NEURONS 2    //Number of 1st layer neurons
#define LAYER2_NEURONS 2    //Number of 2nd layer neurons
#define DEBUG 2 //Debugging options


double WL1[LAYER1_NEURONS][INPUT_SIZE+1];       //The weights of the 1st layer
double WL2[LAYER2_NEURONS][LAYER1_NEURONS+1];   //The weights of the 2nd layer

double DL1[LAYER1_NEURONS]; //The inner states of 1st layer
double DL2[LAYER2_NEURONS]; //The inner states of 2nd layer

double OL1[LAYER1_NEURONS]; //The outer states of 1st layer
double OL2[LAYER2_NEURONS]; //The outer states of 2nd layer



/**
 * Calculates the inner state of all neurons <i> of a given layer
 **/
void CalculateInnerStates(double *Inputs, double *InnerStates, double *Weights, int InputSize, int OutputSize) {
    if (DEBUG==4) printf("Now calculating a layer's outer states...\n");
    for (int i=0; i<OutputSize; i++) {
        InnerStates[i] = Weights[i*InputSize + InputSize]; //The "unity" neuron
        for (int j=0; j<InputSize; j++) {
            InnerStates[i] += Inputs[j] * Weights[i*InputSize + j];
        }
    }
}


/**
 * Calculates the outer state of all neurons <i> of a given layer
 **/
void CalculateOuterStates(double *InnerStates, double *OuterStates, int OutputSize) {
    if (DEBUG==4) printf("Now calculating a layer's outer states...\n");
    for (int i=0; i<OutputSize; i++) {
        OuterStates[i] = 1/(1+exp(-InnerStates[i]));
    }
}


/**
 * Activates a specific layer
 **/
void ActivateLayer(double *Inputs, double *InnerStates, double *Outputs, double *Weights, int InputSize, int OutputSize) {
    if (DEBUG==3) printf("Now calculating a layer's output...\n");
    CalculateInnerStates(Inputs, InnerStates, Weights, InputSize, OutputSize);
    CalculateOuterStates(InnerStates, Outputs, OutputSize);
}


/**
 * Activates the whole Neural Network
 **/
void ActivateNeuralNetwork(double *Inputs) {
    if (DEBUG==2) printf("Now calculating the network's output...\n");
    ActivateLayer(Inputs, DL1, OL1, &WL1[0][0], INPUT_SIZE, LAYER1_NEURONS);
    ActivateLayer(Inputs, DL2, OL2, &WL2[0][0], LAYER1_NEURONS, LAYER2_NEURONS);
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
    InitializeLayerWeights(&WL1[0][0], LAYER1_NEURONS, INPUT_SIZE+1);
    InitializeLayerWeights(&WL2[0][0], LAYER2_NEURONS, LAYER1_NEURONS+1);
}


/**
 * The main program
 **/
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("This program implements a Neural Network of %d Layers.\n", 2);
	printf("Inputs: %d, Hidden layer neurons: %d, Output layer neurons: %d\n", INPUT_SIZE, LAYER1_NEURONS, LAYER2_NEURONS);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");


    double Inputs[INPUT_SIZE] = {1,1};
    InitializeAllWeights();
    ActivateNeuralNetwork(Inputs);
    
    printf("\nThe input was:\n");
    for (int i=0; i<INPUT_SIZE; i++) printf("%lf\n", Inputs[i]);

    printf("\nThe weights of the hidden layer are:\n");
    for (int i=0; i<LAYER1_NEURONS; i++) 
        for (int j=0; j<INPUT_SIZE+1; j++)
            printf("%lf\n", WL1[i][j]);

    printf("\nThe weights of the output layer are:\n");
    for (int i=0; i<LAYER2_NEURONS; i++) 
        for (int j=0; j<LAYER1_NEURONS+1; j++)
            printf("%lf\n", WL2[i][j]);

    printf("\nThe output was:\n");
    for (int i=0; i<LAYER2_NEURONS; i++) printf("%lf\n", OL2[i]);


    return 0;
}