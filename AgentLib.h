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
const int population = 1<<10;
const int windowSize = 1<<5;
//sexual Identity 
typedef enum sexualIdentityE{
    hetero,
    MSM,
    bi
}sexualIdentity;

//type of relationship
typedef enum relationshipTypeT{
    casual,
    steady
}relationshipType;

//type of sexual act
typedef enum intercourseT{
    anal,
    fellatio,
    vaginal
}intercourse;

//Disease progrssion state
typedef enum progressE{
    Acute, //of short duration
    F0, //no fibrosis
    F1, //portal fibrosis with no septa
    F2, //portal fibrosis with few septa
    F3, //portal fibrosis with numerous septa
    F4, //compensated cirrohsis
    LF, //liver failure
    HCC, //Hepatocellular carcinoma
    LT  //liver transplant
}progression;

//Drug Injection Behavior
typedef struct dibT{
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

//disease status
typedef struct dsT{
    //infection status
    bool isInfected;

    //genotype    *
    string genoType;

    //viral load     *
    float viralLoad;

    //disease progression
    progression state;

}ds;

//sexual behavior 
typedef struct sbT{
    //type of sexual act    *
    string action;
    
    //sexual activity rate
    float ActivityRate;

    //condom usage rate
    float condomUse;
    
    //number of sexual partners
    float numPartners;

    //type of relationship    *
    relationshipType relationship;

}sb;

/*
network probability parameters
*/

//heterosexual network
typedef struct heteroNetworkProbT{
    //members of this network can hold 1k people may need to be reduced 
    int members[1<<10];
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
    float sexCausal;

    //rate of sexual intercourse for stable partner
    float sexSteady;

    //probability of transmission through sexual network
    float PdiseaseRisk;

    //distance constraint for sexual partnerships
    float distSex;
}heteroNetworkProb;

//MSM network
typedef struct MSMNetworkProbT{
    //members of this network can hold 1k people may need to be reduced 
    int members[1<<10];
    
    //MSM scale free gamma
    float gamma;

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
    float sexRate;

    //probability of transmission through MSM network
    float PdiseaseRisk;

    //distance constraint for sexual partnerships
    float distSex;
}MSMNetworkProb;

//injection network
typedef struct injectionNetworkProbT{
    //IDU = injection drug user

    //members of this network can hold 1k people may need to be reduced 
    int members[1<<10];

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
}injectionNetworkProb;

//define the agent
typedef struct agentT{
    // age of person
    float age;

    //biological sex
    string sexType;

    //sexual id
    sexualIdentity sexualId;

    //immigration status
    bool isImmigraant;

    //sexual behavior
    sb sex;

    //drug injection behavior
    dib drugs;

    //disease status
    ds disease;

    //hetero network
    heteroNetworkProb hNet;

    //MSM network
    MSMNetworkProb MSMNet;

    //injection network 
    injectionNetworkProb inNet;
    
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
    float sexCausal[2];

    //rate of sexual intercourse for stable partner
    float sexSteady[2];

    //probability of transmission through sexual network
    float PdiseaseRisk[2];

    //distance constraint for sexual partnerships
    float distSex[2];
}heteroNetwork;

//MSM network
typedef struct MSMNetworkT{
        //MSM scale free gamma
    float gamma[2];

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

    // rate of sexual intercourse 
    float sexRate[2];

    //probability of transmission through MSM network
    float PdiseaseRisk[2];

    //distance constraint for sexual partnerships
    float distSex[2];
}MSMNetwork;

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
}injectionNetwork;

extern heteroNetwork hn;
extern MSMNetwork msm;
extern injectionNetwork in;

//kernel prototypes
//random kernel setup
__global__ void setupKernel(curandState *state,double seed);

//random uniform number generator kernel 
__global__ void generateUniformKernel(curandState *state, double *result);

//choose agent based on normal distibution with mean at geographical location of current agent
__global__ void chooseRandAgent(curandState *state,double *xCoord, double *yCoord, float std, int offset);

//generate seed for random number
double generateSeed(curandState *state);


#endif