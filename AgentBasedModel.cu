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

using namespace std;

/*
TODO define all probabilites ranges for each network
*/
static heteroNetwork* newhn = {&hn};

int main(int argc, char* argv[]){
    printf("%0.2f\n", newhn->gamma[0]);
    string word = "hello world";
    printf("%s\n", word.c_str());

}