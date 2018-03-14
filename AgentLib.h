#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>
#include <curand_kernel.h>



using namespace std;
#ifndef AGENTLIB_H
#define AGENTLIB_H
/*
    "*" denotes string might be turned into a struct or enum to simplify
*/

//number of agents 
//population needs to be raised to an even power to make calculations eaiser
//const int population = 1<<8;
//const int windowSize = 1<<2;

const int population = 1<<20;
const int windowSize = 1<<10;
const int NearDistance = 10;
const int PDensity = 4; 

//maximum number of links for all network
const int decayRate = 7;
const int maxLinksHetero = population * 0.15;
const int maxLinksMSM = population * 0.005;
const int maxLinksIDU = population * 0.001;
const int linkDecayHetero = maxLinksHetero * 0.1;
const int linkDecayMSM = maxLinksMSM * 0.1;
const int linkDecayIDU = maxLinksMSM * 0.3;

//sexual network params
const float c = .75;
const double lambda = 1.2;

//Drug Injection Behavior
typedef struct dibT{
    bool isUser;
    //injection frequency
    float freq;
    
    //rate of sharing injection equipment
    float shareRate;

    //number of injection partners
    int numPartners;    
}dib;

/*
these need better descriptors
*/

//type of network being evaluated
typedef enum networkT{
    MSM,
    hetero,
    IDU
}network;

//disease status
typedef struct dsT{
    //infection status
    bool isInfected;

    //genotype    *
    char genoType[20];

    //viral load     *
    float viralLoad;

    //disease progression
    char state[10];

}ds;

//sexual behavior 
typedef struct sbT{
    //type of sexual act  {anal, vaginal, other}
    char action[7];
    
    //sexual activity rate
    float ActivityRate;

    //condom usage rate
    float condomUse;
    
    //number of sexual partners
    int numPartners;

    //type of relationship   {steady, casual}
    char relationship[6];

}sb;

/*
network probability parameters
*/

//heterosexual network
typedef struct heteroNetworkProbT{

    //sexual network scale free gamma
    float gamma;

    //probability of looking for a sexual partner
    float Plook;

    //probability of casual partnership
    float Pcasual;

    //duration steady relation ship (months)
    float durSteady;

    //duration casual partnership (months)
    float durCasual;

    //probability of safe sex practice for stable partner
    float PcondomSteady;

    //probability of safe sex practice for casual partner
    float PcondomCasual;

    // rate of sexual intercourse for casual partner
    float sexCasual;

    //rate of sexual intercourse for stable partner
    float sexSteady;

    //probability of transmission through sexual network
    float PdiseaseRisk;

    //distance constraint for sexual partnerships
    float distSex;
}heteroNetProb;

//MSM network
typedef struct MSMNetworkProbT{

    //MSM scale free gamma
    float gamma;

     //probability of looking for a sexual partner
     float Plook;

    //probability of casual partnership
    float Pcasual;

    //duration steady relation ship (months)
    float durSteady;

    //duration casual partnership (months)
    float durCasual;

    //probability of safe sex practice for stable partner
    float PcondomSteady;

    //probability of safe sex practice for casual partner
    float PcondomCasual;

    // rate of sexual intercourse 
    float sexCasual;

    float sexSteady;

    //probability of transmission through MSM network
    float PdiseaseRisk;

    //distance constraint for sexual partnerships
    float distSex;
}MSMNetProb;

//injection network
typedef struct injectionNetworkProbT{
    //IDU = injection drug user

    //average size of scaling group
    float gamma;

    //proportion of injections that are shared for persons that share syringes
    float Pshare;

    //probability of separate of an injection relationship
    float Pseperate;

    //proportion of shared syringes are cleaned before use
    float Pclean;

    //max proportion of IDU in terms of overall population
    float maxLink;

    //rate of NON IDU to occasional IDU
    float useRateRare;

    //rate of occasional IDU regular
    float useRateRegular;

    //rate of occasional to quit
    float useRateQuit;

    //frequency of injection (monthly)
    float injectRateCasual;

    //frequency of injection (daily)
    float injectRateRegular;

    //probability of transmission through IDU network
    float PdiseaseRisk;

    //distance constraint for IDU partnerships
    float distIDU;
}IDUNetProb;

//define the agent
typedef struct agentT{
    // age of person
    float age;

    //biological sex {Male, Female}
    char sexType[6];

    //sexual id {hetero, MSM, bi}
    char sexualId[6];

    //immigration status
    bool isImmigraant;
    
    //number of sexual partners
    int numPartnersSex;

    //sexual behavior
    sb behaviorSex;

    //drug injection behavior
    dib drugs;

    //disease status
    ds disease;

    //hetero network
    heteroNetProb hNet;

    //MSM network
    MSMNetProb MSMNet;

    //injection network 
    IDUNetProb IDUNet;
    
} agent;

/*
//network probabilities structure
These structs hold the ranges of probabilities that will be used to calculate a single probability for each agent
*/

//heterosexual network
typedef struct heteroNetworkT{
    //sexual network scale free gamma
    float gamma[2];

    //probability of looking for a sexual partner
    float Plook[2];

    //probability of casual partnership
    float Pcasual[2];

    //duration steady relation ship (months)
    float durSteady[2];

    //duration casual partnership (months)
    float durCasual[2];

    //probability of safe sex practice for stable partner
    float PcondomSteady[2];

    //probability of safe sex practice for casual partner
    float PcondomCasual[2];

    // rate of sexual intercourse for casual partner
    float sexCasual[2];

    //rate of sexual intercourse for stable partner
    float sexSteady[2];

    //probability of transmission through sexual network
    float PdiseaseRisk[2];

    //distance constraint for sexual partnerships
    float distSex[2];
}heteroNetRanges;

//MSM network
typedef struct MSMNetworkT{
        //MSM scale free gamma
    float gamma[2];

    float Plook[2];
    //probability of casual partnership
    float Pcasual[2];

    //duration steady relation ship (months)
    float durSteady[2];

    //duration casual partnership (months)
    float durCasual[2];

    //probability of safe sex practice for stable partner
    float PcondomSteady[2];

    //probability of safe sex practice for casual partner
    float PcondomCasual[2];

    // rate of sexual intercourse for casual partner
    float sexCasual[2];

    //rate of sexual intercourse for stable partner
    float sexSteady[2];

    //probability of transmission through MSM network
    float PdiseaseRisk[2];

    //distance constraint for sexual partnerships
    float distSex[2];
}MSMNetRanges;

//injection network
typedef struct injectionNetwortT{
    //IDU = injection drug user

        //average size of scaling group
    float gamma[2];

    //proportion of injections that are shared for persons that share syringes
    float Pshare[2];

    //probability of separate of an injection relationship
    float Pseperate[2];

    //proportion of shared syringes are cleaned before use
    float Pclean[2];

    //max proportion of IDU in terms of overall population
    float maxLink[2];

    //rate of NON IDU to occasional IDU
    float useRateRare[2];

    //rate of occasional IDU regular
    float useRateRegular[2];

    //rate of occasional to quit
    float useRateQuit[2];

    //frequency of injection (monthly)
    float injectRateCasual[2];

    //frequency of injection (daily)
    float injectRateRegular[2];

    //probability of transmission through IDU network
    float PdiseaseRisk[2];

    //distance constraint for IDU partnerships
    float distIDU[2];
}IDUNetRanges;

//define isEdge for thrust counting
typedef struct isEdgeT{
    __host__ __device__
    bool operator()(int2 edge){
        if (edge.x == -1 && edge.y == -1){
            return false;
        }
        return true;
    }
}isEdge;

//define struct to remove all non edges
typedef struct compressNetT{
    __host__ __device__
    bool operator () (int2 num){
        if (num.x == -1 && num.y == -1){
            return false;
        }
        return true;
    }
}compressNet;

//define struct to remove all -1
typedef struct compressUserT{
    __host__ __device__
    bool operator () (int num){
        if (num == -1){
            return false;
        }
        return true;
    }
}compressUser;

typedef struct isInfectedT{
    __host__ __device__
    bool operator()(agent person){
        return person.disease.isInfected;
    } 
}infected;

typedef struct countUserT{
    __host__ __device__
    bool operator()(int person){ 
        return person != -1;
    } 
}countUser;

extern heteroNetRanges hnr;
extern MSMNetRanges msmr;
extern IDUNetRanges idur;

/*************************************************************
kernel prototypes
**************************************************************/

//random kernel setup
__global__ void setupKernel(curandState *state,double seed);

//random uniform number generator kernel 
__global__ void generateUniformKernel(curandState *state, double *result);

//choose agent based on normal distibution with mean at geographical location of current agent
__global__ void chooseRandAgent(curandState *state,int *randAgent, network type, int* users, int numUsers);

//populate georaphical list of agents
__global__ void agentsInit(curandState *state, agent* agentListP, heteroNetRanges* hnrD, MSMNetRanges* msmrD, IDUNetRanges* idurD,bool* checkInfected, int* users);

//fills the hetero network
__global__ void fillHeteroNet(curandState *state,int2* hNetEdges, agent* agentList, int2* Male, int2* Female, int* coordD);

//fill the MSM network
__global__ void fillMSMNet(curandState *state, int2* MSMNetEdges, agent* agentList, int2* Male, int* coordD);

//fill the IDU Network
__global__ void fillIDUNet(curandState *state, int2* IDUNetEdges, agent* agentList, int* coordD, int* users, int numUsers);

//trims a network based on max links 
__global__ void trimNet(curandState *state, int2* netEdges, int2* dummy, int maxLinks, int numEdges);

//updates the infected population
__global__ void updateInfections(curandState *state, agent* agentList, int2* Edges, network type);

//remove links after a certain timestep
__global__ void removeLinks(curandState *state, int2* Edges, int numEdges, network type);

//assign people with random locations
__global__ void randomPos(curandState *state,int2 *Male, int2 *Female);

__global__ void random(int2 *Male, int2 *Female,int2* Coed,unsigned int seed, int dist);
/*************************************************************
generic methods
*************************************************************/

//generate seed for random number
double generateSeed();


#endif