#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include "AgentLib.h"
#include <thrust/fill.h>


using namespace std;

// generate seed for curand
extern double generateSeed();

//curand kernel setup
extern __global__ void setupKernel(curandState *state, double seed);

//uniform random kernel
extern __global__ void generateUniformKernel(curandState *state, double *result);

//choose agent iwth normal distributions
extern __global__ void chooseRandAgent(curandState *state,double *xCoord,double *yCoord, float std, int offset);

//select a random agent to calculate
extern void chooseRandAgent(curandState* states);

/*
TODO
uniformily distribute males and females in a list of len(population)
this way when agents are choosen at random then it is equally likely they can be added to either MSM or hetero networks
*/

int main(int argc, char* argv[]){
    double seed; 
    int n = 0;
    double *xCoord;
    double *yCoord;
    
    cudaMallocManaged(&xCoord,sizeof(double)* windowSize* windowSize);
    cudaMallocManaged(&yCoord,sizeof(double)* windowSize* windowSize);
    
    //initialize seed and allcoate space for random number values
    curandState * states;
    cudaMallocManaged(&states,sizeof(curandState)* population );
    seed = generateSeed();
    setupKernel<<<1024,1024>>>(states, seed);
    cudaDeviceSynchronize();

    //test 
    //generate random numbers
    // generateUniformKernel<<<64,64>>>(states,results);
    // cudaDeviceSynchronize();
    
    //sliding window
    //while(links < maxLinks)
    while (windowSize * windowSize * n < population){
        chooseRandAgent<<<windowSize,windowSize>>>(states,xCoord, yCoord, 3.0, n);
        cudaDeviceSynchronize();
        for (int i = 0; i < windowSize* windowSize; i++){
            printf("[%d, %d]\n",(int)xCoord[i], (int)yCoord[i]);
        }
        n++;
    } 
    /*
    
    TODO calculate probability that agentI and agentJ will be added to a network
    increment the number of links
    do this for all networks
    */

    printf("%.f\n", seed);
    string word = "hello world";
    printf("%s\n", word.c_str());


    cudaFree(states);
    cudaFree(xCoord);
    cudaFree(yCoord);

}

