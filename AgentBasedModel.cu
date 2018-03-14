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
#include <thrust/count.h>
#include <thrust/execution_policy.h>

using namespace std;

int main(int argc, char* argv[]){
    double seed;
    int numInfected = 0;
    int timeStep = 0;
    int popWidth = sqrt(population);
    int numLinksHetero = 0;
    int numLinksMSM = 0;
    int numLinksIDU = 0;
    int numUsers = 0;
    int* users;
    int* coordD;
    int* dummyInts;
    agent* agentList;
    heteroNetRanges* hnrH = {&hnr};
    MSMNetRanges* msmrH = {&msmr};
    IDUNetRanges* idurH = {&idur};
    heteroNetRanges* hnrD;
    MSMNetRanges* msmrD;
    IDUNetRanges* idurD;
    int2 *hNetEdges;
    int2 *MSMNetEdges;
    int2 *IDUNetEdges;
    int2* dummyList;
    int2 *Male;
    int2 *Female;
    int2 *Coed;
    bool* isInfected;
   
    cudaMallocManaged(&agentList,sizeof(agent)*population);
    cudaMallocManaged (&Male , popWidth*sizeof(int2));
    cudaMallocManaged (&Female , popWidth*sizeof(int2));
    cudaMallocManaged (&Coed , population *sizeof(int2));
    cudaMallocManaged (&dummyInts , population *sizeof(int));
    //initialize seed and allcoate space for random number values
    curandState * states;
    cudaMallocManaged(&states,sizeof(curandState)* population );
    seed = generateSeed();
    setupKernel<<<1024,1024>>>(states, seed);
    cudaDeviceSynchronize();

    //use sliding window to initialize all agent values
    cudaMallocManaged(&hnrD, sizeof(heteroNetRanges));
    cudaMallocManaged(&msmrD, sizeof(MSMNetRanges));
    cudaMallocManaged(&idurD, sizeof(IDUNetRanges));
    cudaMallocManaged(&users, sizeof(int)*population);
    cudaMallocManaged(&isInfected,sizeof(bool)* population);


    cudaMemcpy(hnrD, hnrH, sizeof(heteroNetRanges), cudaMemcpyHostToDevice);
    cudaMemcpy(msmrD, msmrH, sizeof(MSMNetRanges), cudaMemcpyHostToDevice);
    cudaMemcpy(idurD, idurH, sizeof(IDUNetRanges), cudaMemcpyHostToDevice);
    
    thrust::fill(isInfected, isInfected+population, false);
    //initialize all agents

    agentsInit<<<windowSize,windowSize>>>(states, agentList, hnrD, msmrD, idurD, isInfected, users);
    cudaDeviceSynchronize();
    numUsers = population - thrust::count(users, users+population, -1);
    
    printf("population %d \nnum users %d\n",population, numUsers);
    random<<<windowSize, windowSize>>>(Male, Female,Coed, time(0), 225);
    cudaDeviceSynchronize();

    //if(thrust::count(isInfected, isInfected+population, true) < population * .0003){
        int max = population * .0001;
        for (int i = 0; i < 500; i++ ){
            int num = rand()% population;
            agentList[num].disease.isInfected = true;
        }
    //}

    thrust::fill(dummyInts, dummyInts+population, -1);
    thrust::copy_if(thrust::device, users, users+population, dummyInts, compressUser());
    numUsers = population - thrust::count(dummyInts, dummyInts + population, -1);
    thrust::fill(users, users + population, -1);
    thrust::copy_n(users,numUsers, dummyInts);
    printf("new num users %d\n", numUsers);

    cudaFree(isInfected);
    cudaFree(hnrD);
    cudaFree(msmrD);
    cudaFree(idurD);

    //create list of edges initilized to 0
    cudaMallocManaged(&hNetEdges, sizeof(int2) * population);
    cudaMallocManaged(&MSMNetEdges, sizeof(int2) * population);
    cudaMallocManaged(&IDUNetEdges, sizeof(int2) * population);
    cudaMallocManaged(&dummyList, sizeof(int2)*population);
    int2 init;
    init.x = -1;
    init.y = -1;
    thrust::fill(hNetEdges, hNetEdges+population, init);
    thrust::fill(MSMNetEdges, MSMNetEdges+population, init);
    thrust::fill(IDUNetEdges, IDUNetEdges+population, init);
    thrust::fill(dummyList, dummyList+population, init);

    //assign a random person to each agent  
    //and fill networks until maxLinks is reached 
    cudaMallocManaged(&coordD,sizeof(int)* population); 
    printf("the number of Max links Hetero = %d\n", maxLinksHetero);
    printf("the number of Max links MSM = %d\n", maxLinksMSM);
    printf("the number of Max links IDU = %d\n", maxLinksIDU);
    while(timeStep  < 5000){
        //addlinks to hetero network
        while(numLinksHetero < maxLinksHetero){
            //choose random agent for every person hetero
            chooseRandAgent<<<windowSize,windowSize>>>(states, coordD, hetero, users, numUsers);
            cudaDeviceSynchronize();
            fillHeteroNet<<<windowSize,windowSize>>>(states, hNetEdges, agentList, Male, Female, coordD);
            cudaDeviceSynchronize();
            numLinksHetero = thrust::count_if(thrust::host,hNetEdges, hNetEdges+population, isEdge());
            thrust::copy_if(thrust::host, hNetEdges, hNetEdges + population, hNetEdges, compressNet());
        }

        //addlinks to MSM network
        while(numLinksMSM < maxLinksMSM){
            //select another agent from the list of males
            chooseRandAgent<<<windowSize,windowSize>>>(states, coordD, MSM,users, numUsers);
            cudaDeviceSynchronize();
            fillMSMNet<<<windowSize,windowSize>>>(states, MSMNetEdges, agentList, Male, coordD);
            cudaDeviceSynchronize();
            numLinksMSM = thrust::count_if(thrust::host,MSMNetEdges, MSMNetEdges+population, isEdge());
            thrust::copy_if(thrust::host, MSMNetEdges, MSMNetEdges + population, MSMNetEdges, compressNet());   
       
            }
        //addlinks to IDU network
        while(numLinksIDU < maxLinksIDU && numLinksIDU < numUsers){
            //select another agent from the list of males
            chooseRandAgent<<<windowSize,windowSize>>>(states, coordD, IDU,users, numUsers);
            cudaDeviceSynchronize();
            fillIDUNet<<<windowSize,windowSize>>>(states, IDUNetEdges, agentList, coordD, users, numUsers);
            cudaDeviceSynchronize();
            numLinksIDU = thrust::count_if(thrust::host,IDUNetEdges, IDUNetEdges+population, isEdge());
            thrust::copy_if(thrust::host, IDUNetEdges, IDUNetEdges + population, IDUNetEdges, compressNet());           
        }

        //if number of links formed is greater than maxLinks allowed remove some
        if(numLinksHetero > maxLinksHetero){
            trimNet<<<windowSize, windowSize>>>(states, hNetEdges, dummyList, maxLinksHetero, numLinksHetero);
            cudaDeviceSynchronize();
            cudaMemcpy(hNetEdges, dummyList, sizeof(int2)* population, cudaMemcpyDeviceToDevice);
            thrust::fill(dummyList, dummyList+population, init);
        }
        if(numLinksMSM > maxLinksMSM){
            trimNet<<<windowSize, windowSize>>>(states, MSMNetEdges, dummyList, maxLinksMSM, numLinksMSM);
            cudaDeviceSynchronize();
            cudaMemcpy(MSMNetEdges, dummyList, sizeof(int2)* population, cudaMemcpyDeviceToDevice);
            thrust::fill(dummyList, dummyList+population, init);
        }

        if(numLinksIDU > maxLinksIDU){
            trimNet<<<windowSize, windowSize>>>(states, IDUNetEdges, dummyList, maxLinksIDU, numLinksIDU);
            cudaDeviceSynchronize();
            cudaMemcpy(IDUNetEdges, dummyList, sizeof(int2)* population, cudaMemcpyDeviceToDevice);
            thrust::fill(dummyList, dummyList+population, init);
            }

        //remove edges from networks if over a certain timestep
        if(timeStep % decayRate == 0){
            removeLinks<<<windowSize, windowSize>>>(states, hNetEdges, numLinksHetero, hetero);
            cudaDeviceSynchronize();
            thrust::copy_if(thrust::host, hNetEdges, hNetEdges + population, hNetEdges, compressNet());
            removeLinks<<<windowSize, windowSize>>>(states, MSMNetEdges, numLinksMSM, MSM);
            cudaDeviceSynchronize();
            removeLinks<<<windowSize, windowSize>>>(states, MSMNetEdges, numLinksIDU, IDU);
            cudaDeviceSynchronize();
            numLinksHetero = thrust::count_if(thrust::host,hNetEdges, hNetEdges+population, isEdge());
            numLinksMSM = thrust::count_if(thrust::host,MSMNetEdges, MSMNetEdges+population, isEdge());
        }
        
        //update the infected population
        updateInfections<<<windowSize, windowSize>>>(states, agentList, hNetEdges, hetero);
        cudaDeviceSynchronize();
        updateInfections<<<windowSize, windowSize>>>(states, agentList, MSMNetEdges, MSM);
        cudaDeviceSynchronize();
        updateInfections<<<windowSize, windowSize>>>(states, agentList, IDUNetEdges, IDU);
        cudaDeviceSynchronize();
        numInfected = thrust::count_if(thrust::host,agentList, agentList+population, infected());
        timeStep++;
        if(timeStep % 50 ==0){
            printf("the number of people infected after %d timestep is %d\n",timeStep, numInfected);
        }
        
    }    
       
    printf("gets here done\n");
    cudaFree(hNetEdges);
    cudaFree(coordD);
    cudaFree(dummyList);
    cudaFree(Male);
    cudaFree(Female);
    cudaFree(agentList);

    printf("%.f\n", seed);
    string word = "hello world";
    printf("%s\n", word.c_str());
    cudaFree(states);
}

