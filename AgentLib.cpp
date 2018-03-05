#include <stdio.h>
#include <stdlib.h>
#include <string>
using namespace std;

/*
    "*" denotes string might be turned into a struct or enum to simplify
*/

//namespace to use in temp lib
namespace Agentlib{

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
    } agent;

    /*
    //network probabilities structure
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

        //frequency of injection (monthly)
        float injectRateCasual[2];

        //frequency of injection (daily)
        float injectRateRegular[2];

        //probability of transmission through IDU network
        float PdiseaseRisk[2];

        //distance constraint for IDU partnerships
        float distIDU[2];
    }injectionNetwork;
}
