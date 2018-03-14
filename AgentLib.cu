#include <stdio.h>
#include <stdlib.h>
#include "AgentLib.h"
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

heteroNetRanges hnr = {
    //ranges of probabilities for hetero network
    {2.0, 4.0},    //gamma
    {0.5, 0.99},   //Plook
    {0.01, 0.3},   //Pcasual
    {20.0, 300.0}, //durSteady
    {3.0, 15.0},   //durCaual
    {0.1, 0.6},    //PcondomSteady
    {0.3, 0.7},    //PcondomCasual
    {1.0, 24.0},   //sexSteady
    {1.0, 26.0},   //sexCasual
    {0.000001, 0.000002}, //PdiseaseRisk
    {50.0, 200.0}  //distSex
};

//ranges of probabilities for MSM network
MSMNetRanges msmr = {
    {1.0, 2.0},        //gamma
    {0.6,0.99},       //Plook
    {0.01, 0.1},      //Pcasual
    {1.0 *12, 2.0*12},//durSteady
    {0.1, 1.0},       //durCaual
    {0.3, 0.7},       //PcondomSteady
    {0.7, 0.99},       //PcondomCasual
    {10.0, 20.0},      //sexSteady
    {18.0, 36.0},      //sexCasual
    {0.000005, 0.0015}, //PdiseaseRisk
    {50.0, 200.0}      //distSex
};

//range of probabilites for injection network
IDUNetRanges idur = {
    {2.0, 4.0},      //gamma
    {0.13, 0.17},    //Pshare;
    {0.2, 0.3},
    {0.0, 0.01},
    {0.01, 0.1},
    {0.05, 0.192},
    {0.05, 0.192},
    {0.44, 0.69},
    {1.0, 3.0},
    {2.0, 5.0},
    {0.0025, 0.005},
    {50.0, 200.0}
};

//define a string string copy for use on device
__device__ char* strcpyD(char* dest, const char *src){
    int i =0; 
    do{
        dest[i] = src[i];
    }while(src[i++] != 0);
    return dest;
}

// //load network probability ranges onto device for initialization
// void loadNetRanges(){
//     cudaMallocManaged(&hnrD, sizeof(heteroNetRanges));
//     cudaMallocManaged(&msmrD, sizeof(MSMNetRanges));
//     cudaMallocManaged(&idurD, sizeof(IDUNetRanges));
//     cudaMemcpy(&hnrD, &hnr, sizeof(heteroNetRanges), cudaMemcpyHostToDevice);
//     cudaMemcpy(&msmrD, &msmr, sizeof(MSMNetRanges), cudaMemcpyHostToDevice);
//     cudaMemcpy(&idurD, &idur, sizeof(IDUNetRanges), cudaMemcpyHostToDevice);
// }

// //free network probability ranges from device
// void freeNetRanges(){
//     cudaFree(hnrD);
//     cudaFree(msmrD);
//     cudaFree(idurD);
// }


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

//choose agent based on uniform distibution within a window area
extern __global__ void chooseRandAgent(curandState *state,int *randAgent, network type, int* users, int numUsers){
    //n is the length of the geographical grid
    int id = threadIdx.x + blockIdx.x * gridDim.x;
    int posGlobal;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    /* Generate pseudo-random normals between window min and window max*/
    if(id < population){
        switch(type){
            case hetero:
                posGlobal = (int)(curand_uniform(&localState) * population);
                while(posGlobal % 2 == id % 2){
                    posGlobal = (int)(curand_uniform(&localState) * population);
                }
                randAgent[id] = posGlobal;
                break;
            case MSM:
                posGlobal = (int)(curand_uniform(&localState) * population);
                while(posGlobal % 2 != id % 2){
                    posGlobal = (int)(curand_uniform(&localState) * population);
                }
                randAgent[id] = posGlobal;
                break;
            case IDU:
                if(users[id] != -1){
                    posGlobal = (int)(curand_uniform(&localState) * numUsers);
                    randAgent[id] = users[posGlobal];
                }

                break;
        }
    
        
    }
    /* Copy state back to global memory */
    state[id] = localState;

}

//Initialize agent values
extern __global__ void agentsInit(curandState *state, agent* agentListP, heteroNetRanges* hnrD, MSMNetRanges* msmrD, IDUNetRanges* idurD,bool* checkInfected, int* users){
    int agentId = threadIdx.x + blockIdx.x*gridDim.x;
    curandState localState = state[agentId];
    //generate random unifor number and use this to initialize all params for networks
    double x = curand_uniform(&localState);
   
    
    //assign males to even positions and females to odd positions
    if(agentId < population){

        if (agentId %2 == 0){
            strcpyD(agentListP[agentId].sexType, "Male");
        }
        else{
            strcpyD(agentListP[agentId].sexType, "Female");
        }
    
        //heteroNet param init
        agentListP[agentId].hNet.gamma = (float)(curand_uniform(&localState) * (hnrD->gamma[1]- hnrD->gamma[0]) + hnrD->gamma[0]);
        agentListP[agentId].hNet.Plook = (float)(curand_uniform(&localState) * (hnrD->Plook[1]- hnrD->Plook[0]) + hnrD->Plook[0]);
        agentListP[agentId].hNet.Pcasual = (float)(curand_uniform(&localState) * (hnrD->Pcasual[1]- hnrD->Pcasual[0]) + hnrD->Pcasual[0]);
        agentListP[agentId].hNet.durSteady = (float)(curand_uniform(&localState) * (hnrD->durSteady[1]- hnrD->durSteady[0]) + hnrD->durSteady[0]);
        agentListP[agentId].hNet.durCasual = (float)(curand_uniform(&localState) * (hnrD->durCasual[1]- hnrD->durCasual[0]) + hnrD->durCasual[0]);
        agentListP[agentId].hNet.PcondomSteady = (float)(curand_uniform(&localState) * (hnrD->PcondomSteady[1]- hnrD->PcondomSteady[0]) + hnrD->PcondomSteady[0]);
        agentListP[agentId].hNet.PcondomCasual = (float)(curand_uniform(&localState) * (hnrD->PcondomCasual[1]- hnrD->PcondomCasual[0]) + hnrD->PcondomCasual[0]);
        agentListP[agentId].hNet.sexCasual = (float)(curand_uniform(&localState) * (hnrD->sexCasual[1]- hnrD->sexCasual[0]) + hnrD->sexCasual[0]);
        agentListP[agentId].hNet.sexSteady = (float)(curand_uniform(&localState) * (hnrD->sexSteady[1]- hnrD->sexSteady[0]) + hnrD->sexSteady[0]);
        agentListP[agentId].hNet.PdiseaseRisk = (float)(curand_uniform(&localState) * (hnrD->PdiseaseRisk[1]- hnrD->PdiseaseRisk[0]) + hnrD->PdiseaseRisk[0]);
        agentListP[agentId].hNet.distSex = (float)(curand_uniform(&localState) * (hnrD->distSex[1]- hnrD->distSex[0]) + hnrD->distSex[0]);

        //MSMNet paramInit
        agentListP[agentId].MSMNet.gamma = (float)(curand_uniform(&localState) * (msmrD->gamma[1]- msmrD->gamma[0]) + msmrD->gamma[0]);
        agentListP[agentId].MSMNet.Plook = (float)(curand_uniform(&localState) * (msmrD->Plook[1]- msmrD->Plook[0]) + msmrD->Plook[0]);
        agentListP[agentId].MSMNet.Pcasual = (float)(curand_uniform(&localState) * (msmrD->Pcasual[1]- msmrD->Pcasual[0]) + msmrD->Pcasual[0]);
        agentListP[agentId].MSMNet.durSteady = (float)(curand_uniform(&localState) * (msmrD->durSteady[1]- msmrD->durSteady[0]) + msmrD->durSteady[0]);
        agentListP[agentId].MSMNet.durCasual = (float)(curand_uniform(&localState) * (msmrD->durCasual[1]- msmrD->durCasual[0]) + msmrD->durCasual[0]);
        agentListP[agentId].MSMNet.PcondomSteady = (float)(curand_uniform(&localState) * (msmrD->PcondomSteady[1]- msmrD->PcondomSteady[0]) + msmrD->PcondomSteady[0]);
        agentListP[agentId].MSMNet.PcondomCasual = (float)(curand_uniform(&localState) * (msmrD->PcondomCasual[1]- msmrD->PcondomCasual[0]) + msmrD->PcondomCasual[0]);
        agentListP[agentId].MSMNet.sexCasual = (float)(curand_uniform(&localState) * (msmrD->sexCasual[1]- msmrD->sexCasual[0]) + msmrD->sexCasual[0]);
        agentListP[agentId].MSMNet.sexSteady = (float)(curand_uniform(&localState) * (msmrD->sexSteady[1]- msmrD->sexSteady[0]) + msmrD->sexSteady[0]);
        agentListP[agentId].MSMNet.PdiseaseRisk = (float)(curand_uniform(&localState) * (msmrD->PdiseaseRisk[1]- msmrD->PdiseaseRisk[0]) + msmrD->PdiseaseRisk[0]);
        agentListP[agentId].MSMNet.distSex = (float)(curand_uniform(&localState) * (msmrD->distSex[1]- msmrD->distSex[0]) + msmrD->distSex[0]);
        
        //IDU paramInit
        agentListP[agentId].IDUNet.gamma = (float)(curand_uniform(&localState) * (idurD->gamma[1]- idurD->gamma[0]) + idurD->gamma[0]);
        agentListP[agentId].IDUNet.Pshare = (float)(curand_uniform(&localState) * (idurD->Pshare[1]- idurD->Pshare[0]) + idurD->Pshare[0]);
        agentListP[agentId].IDUNet.Pseperate = (float)(curand_uniform(&localState) * (idurD->Pseperate[1]- idurD->Pseperate[0]) + idurD->Pseperate[0]);
        agentListP[agentId].IDUNet.Pclean = (float)(curand_uniform(&localState) * (idurD->Pclean[1]- idurD->Pclean[0]) + idurD->Pclean[0]);
        agentListP[agentId].IDUNet.useRateRare = (float)(curand_uniform(&localState) * (idurD->useRateRare[1]- idurD->useRateRare[0]) + idurD->useRateRare[0]);
        agentListP[agentId].IDUNet.useRateRegular = (float)(curand_uniform(&localState) * (idurD->useRateRegular[1]- idurD->useRateRegular[0]) + idurD->useRateRegular[0]);
        agentListP[agentId].IDUNet.useRateQuit = (float)(curand_uniform(&localState) * (idurD->useRateQuit[1]- idurD->useRateQuit[0]) + idurD->useRateQuit[0]);
        agentListP[agentId].IDUNet.injectRateCasual = (float)(curand_uniform(&localState) * (idurD->injectRateCasual[1]- idurD->injectRateCasual[0]) + idurD->injectRateCasual[0]);
        agentListP[agentId].IDUNet.injectRateRegular = (float)(curand_uniform(&localState) * (idurD->injectRateRegular[1]- idurD->injectRateRegular[0]) + idurD->injectRateRegular[0]);
        agentListP[agentId].IDUNet.PdiseaseRisk = (float)(curand_uniform(&localState) * (idurD->PdiseaseRisk[1]- idurD->PdiseaseRisk[0]) + idurD->PdiseaseRisk[0]);
        agentListP[agentId].IDUNet.distIDU = (float)(curand_uniform(&localState) * (idurD->distIDU[1]- idurD->distIDU[0]) + idurD->distIDU[0]);
        //sexual behavior
        int p = curand_poisson(&localState, lambda);
        agentListP[agentId].behaviorSex.numPartners = p;
        p = curand_poisson(&localState, lambda);
        agentListP[agentId].drugs.numPartners = p;
    
        //disease status
        if(agentListP[agentId].hNet.PdiseaseRisk > 1 - x ){
            agentListP[agentId].disease.isInfected = true;
            checkInfected[agentId] =true;

        }
        else{
            agentListP[agentId].disease.isInfected = false;
        }
        
        //drug user
        if (x < .001){
            agentListP[agentId].drugs.isUser = true;
            users[agentId] = agentId;
        }
        else{
            agentListP[agentId].drugs.isUser = false;
            users[agentId] = -1;
        }

        /* Copy state back to global memory */

    }
    state[agentId] = localState;
}

//fills the hetero network
extern __global__ void fillHeteroNet(curandState *state, int2* hNetEdges, agent* agentList, int2* Male, int2* Female, int* coordD){
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    double pForm;
    curandState localState = state[id];
    if(id < population){
        int randPos = coordD[id];
        int num = (id%2) == 0 ? id/2 : randPos / 2; 
        float posA = hypotf(Male[num].x, Male[num].y) ;
        num = (id%2) == 1 ? id/2 : randPos / 2;
        float posB = hypotf(Female[num].x, Female[num].y);
        float dist = hypotf(posA, posB);
        double x = curand_uniform(&localState);

        //calculate proability to add edge
        if (dist < max(agentList[id].hNet.distSex,agentList[id].hNet.distSex)){
            pForm = agentList[id].hNet.Plook * agentList[id].hNet.Plook 
                            * (1 - c)
                            +((c * agentList[id].behaviorSex.numPartners
                            * agentList[id].behaviorSex.numPartners)
                            / maxLinksHetero);
        }
        else{
            pForm =  1.0 / 100 * agentList[id].hNet.Plook * agentList[id].hNet.Plook 
                            * (1 - c)
                            +((c * agentList[id].behaviorSex.numPartners
                            * agentList[id].behaviorSex.numPartners)
                            / maxLinksHetero);
        }
        
        if (pForm > x){
            hNetEdges[id].x = id;
            hNetEdges[id].y = randPos;
        }

    }
    state[id] = localState;
    
}

//fill the MSM network
extern __global__ void fillMSMNet(curandState *state, int2* MSMNetEdges, agent* agentList, int2* Male, int* coordD){
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    double pForm;
    curandState localState = state[id];
    if(id < population){
        int randPos = coordD[id];
        int num = id/2; 
        float posA = hypotf(Male[num].x, Male[num].y) ;
        float posB = hypotf(Male[randPos / 2].x, Male[randPos/2].y);
        float dist = hypotf(posA, posB);
        double x = curand_uniform(&localState);

        //calculate proability to add edge
        if (dist < max(agentList[id].MSMNet.distSex,agentList[id].MSMNet.distSex)){
            pForm = agentList[id].MSMNet.Plook * agentList[id].MSMNet.Plook 
                            * (1 - c)
                            +((c * agentList[id].behaviorSex.numPartners
                            * agentList[id].behaviorSex.numPartners)
                            / maxLinksMSM);
        }
        else{
            pForm =  1.0 / 100 * agentList[id].MSMNet.Plook * agentList[id].MSMNet.Plook 
                            * (1 - c)
                            +((c * agentList[id].behaviorSex.numPartners
                            * agentList[id].behaviorSex.numPartners)
                            / maxLinksMSM);
        }
        
        if (pForm > x){
            MSMNetEdges[id].x = id;
            MSMNetEdges[id].y = randPos;
        }

    }
    state[id] = localState;
    
}

//fill the IDU network
extern __global__ void fillIDUNet(curandState *state, int2* IDUNetEdges, agent* agentList, int* coordD, int* users, int numUsers){
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    double pForm;
    curandState localState = state[id];
    if(id < numUsers){
        int randUser = coordD[id]; 
        int user = users[id];
        double x = curand_uniform(&localState);
        double dist = curand_uniform(&localState) * 250;
        float num = max(agentList[id].IDUNet.distIDU,agentList[id].IDUNet.distIDU);
        //calculate proability to add edge
        if (dist <num ){
            pForm =  (1 - c)
                    +((c * agentList[id].drugs.numPartners
                    * agentList[id].drugs.numPartners)
                    / maxLinksIDU);
        }
        else{
            pForm =  1.0 / 100 * (1 - c)
                    +((c * agentList[id].drugs.numPartners
                    * agentList[id].drugs.numPartners)
                    / maxLinksIDU);
        }
        
        if (pForm > x){
            IDUNetEdges[id].x = user;
            IDUNetEdges[id].y = randUser;
        }

    }
    state[id] = localState;
    
}

//sorts a network and randomly selects edges if too many exist
extern __global__ void trimNet(curandState *state, int2* netEdges, int2* dummy, int maxLinks, int numEdges){
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[id];
    
    for (int i = id; i < maxLinks; i += stride){
        int index = (int)(curand_uniform(&localState) * numEdges);
        dummy[i] = netEdges[index];
    }
     
    state[id] = localState;

}

//updates population infections
__global__ void updateInfections(curandState *state, agent* agentList, int2* Edges, network type){
    int id = threadIdx.x + blockIdx.x*gridDim.x;
    curandState localState = state[id];
    
    double x = curand_uniform(&localState);
    double pTransmit = 0.0;
    double pDR = 0.9999;
    double exponent = 1;

    if (id < population){
        int agent1 = Edges[id].x;
        int agent2 = Edges[id].y;
        if(agentList[agent1].disease.isInfected || agentList[agent2].disease.isInfected){
            switch(type){
                case MSM:
                    pDR = max(agentList[agent1].MSMNet.PdiseaseRisk, agentList[agent2].MSMNet.PdiseaseRisk);
                    exponent = max(agentList[agent1].MSMNet.sexCasual,agentList[agent2].MSMNet.sexCasual);
                    break;
                case hetero:
                    pDR = max(agentList[agent1].hNet.PdiseaseRisk, agentList[agent2].hNet.PdiseaseRisk);
                    exponent = max(agentList[agent1].hNet.sexCasual,agentList[agent2].hNet.sexCasual);
                    break;
                case IDU:
                    pDR = max(agentList[agent1].IDUNet.PdiseaseRisk, agentList[agent2].IDUNet.PdiseaseRisk);
                    exponent = max(agentList[agent1].IDUNet.injectRateRegular,agentList[agent2].IDUNet.injectRateRegular);
                    break;
            }
            
            //calculate probability of infection
            pTransmit = 1 - powf(1-pDR, exponent);
            if(pTransmit > x){
                agentList[agent1].disease.isInfected = true;
                agentList[agent2].disease.isInfected = true;
            }
        }
        
    }
    state[id] = localState;
}


//remove links from a network
__global__ void removeLinks(curandState *state, int2* Edges, int numEdges, network type){
    int id = threadIdx.x + blockIdx.x*gridDim.x;

    curandState localState = state[id];
    int index = curand_uniform(&localState) * numEdges;
    int decay;

    switch(type){
        case hetero:
            decay = linkDecayHetero;
            break;
        case MSM:
            decay = linkDecayMSM;
            break;
        case IDU:
            decay = linkDecayIDU; 
            break;
    }
    
    if(id < decay){
        Edges[index].x = -1;
        Edges[index].y = -1;
    }
    state[id] = localState;
}

extern __global__ void random(int2 *Male, int2 *Female, int2* Coed, unsigned int seed, int dist) {

    int index =  blockIdx.x*blockDim.x + threadIdx.x;
    curandState_t state;
  
    /* we have to initialize the state */
    curand_init(index+seed, /* the seed controls the sequence of random values that are produced */
                index, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);
  
    /* curand works like rand - except that it takes a state as a parameter */
    if(index < population / 2){
        Male[index].x = (unsigned int)curand(&state) % dist;
        Male[index].y = (unsigned int)curand(&state) % dist;
        Female[index].x = (unsigned int)curand(&state) % dist;
        Female[index].y = (unsigned int)curand(&state) % dist;
        Coed[index].x = (unsigned int)curand(&state) % dist;
        Coed[index].x = (unsigned int)curand(&state) % dist;
    }
    else{
        Coed[index].x = (unsigned int)curand(&state) % dist;
        Coed[index].x = (unsigned int)curand(&state) % dist;
    }
    

  }