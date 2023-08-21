#include <time.h>
#include <stdio.h>
#include "mt19937ar.h"
#define  COUNT  200

int main() {
      
	init_genrand(time(0));
       	
	printf("%ld", genrand_int31());
	
	return 0;
}
