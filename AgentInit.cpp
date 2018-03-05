#include <stdio.h>
#include <stdlib.h>
#include "AgentLib.h"

using namespace std;

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
