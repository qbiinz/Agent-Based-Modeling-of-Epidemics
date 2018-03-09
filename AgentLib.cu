#include <stdio.h>
#include <stdlib.h>
#include "AgentLib.h"
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;
//int population;
//int windowSize;

heteroNetwork hn = {
    //ranges of pr  obabilities for hetero network
    {2.0, 4.0},
    {0.5, 0.99},
    {0.01, 0.3},
    {20.0, 300.0},
    {3.0, 15.0},
    {0.1, 0.6},
    {0.3, 0.7},
    {1.0, 24.0},
    {0.000001, 0.000002},
    {50.0, 200.0}
};

//ranges of probabilities for MSM network
MSMNetwork msm = {
    {1.0, 2.0},
    {0.01, 0.1},
    {1.0, 2.0},
    {0.1, 1.0},
    {0.3, 0.7},
    {0.7, 0.99},
    {10.0, 20.0},
    {0.000005, 0.0015},
    {50.0, 200.0}
};

//range of probabilites for injection network
injectionNetwork in = {
    {2.0, 4.0},
    {0.13, 0.17},
    {0.2, 0.3},
    {0.0, 0.01},
    {0.01, 0.1},
    {0.05, 0.192},
    {0.05, 0.192},
    {0.44, 0.69},
    {1.0, 3.0},
    {2.0, 5.0},
    {0.025, 0.05},
    {50.0, 200.0}
};

// generate seed for curand
extern double generateSeed(){
    time_t timer;
    struct tm y2k = {0};
    double seconds;

    y2k.tm_hour =0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    time(&timer);
    seconds = difftime(timer, mktime(&y2k));
    return seconds;

}

//curand kernel setup
extern __global__ void setupKernel(curandState *state, double seed)
{
    //change 64 to blockDim.x in future
    int id = threadIdx.x + blockIdx.x * gridDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i =0; i < population; i += stride){
        curand_init(seed, id, 0, &state[id]);
    }
    /* Each thread gets same seed, a different sequence 
       number, no offset */

}

//uniform random kernel
extern __global__ void generateUniformKernel(curandState *state, double *result){
    int id = threadIdx.x + blockIdx.x * 64;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    /* Store results */
    double x;
    x = curand_uniform(&localState);

    result[id] = x;

    /* Copy state back to global memory */
    state[id] = localState;
    
} 

//choose agent based on mormal distibution with mean at geographical location of current agent
extern __global__ void chooseRandAgent(curandState *state,double *xCoord, double *yCoord, float std,int offset){
    //n is the length of the geographical grid
    float end = windowSize - 1;
    int agentId = threadIdx.x + blockIdx.x * gridDim.x + (windowSize * windowSize * offset);
    int resultId = threadIdx.x + blockIdx.x * gridDim.x;
    int x = agentId / windowSize;
    int y = agentId % windowSize;
    int randX = x;
    int randY = y;
    /* Copy state to local memory for efficiency */
    curandState localState = state[agentId];

    /* Generate pseudo-random normals between 0 and window*/
    while (randX == x || randY == y){
        x = max(min((curand_normal(&localState) * std) + x, end),0.0);
        y = max(min((curand_normal(&localState) * std) + y, end),0.0);
    } 
    
    xCoord[resultId] = x;
    yCoord[resultId] = y;

    /* Copy state back to global memory */
    state[agentId] = localState;

}

