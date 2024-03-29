#ifndef MASTER_H_
#define MASTER_H_

#include "kmeans.h"
#include "triple.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <limits>

class KMEANS;

class MASTER
{
public:
    MASTER(char*input, int k, int dimension, int iterations, char*initcenter);
    MASTER(char*input, int k, int dimension, int iterations);
    MASTER(char*input, int k, int dimension, int iterations, bool projections);
    virtual~MASTER();
    double***run(size_t repetitions);
    bool readFile(char*filename);
    double calccosts();
    void readInitialCenters(char*initcenter);
private:
    double squaredDistance(double*p1, double*p2);
    int k;
    int numberOfPoints;
    int dimension;
    double**centers;
    int iterations;
    double costs;
    double**finalcenters;
    bool projections;

    //first entry weight, second entry vector containing the sum of all coordiantes in every
    //dimension, third entry center membership (int from 0 to k-1, initally set to -1)
    triple<double, double*, int>** points;
};

MASTER::MASTER(char*input, int k, int dimension, int iterations, char*initcenter) :
dimension(dimension),
k(k),
iterations(iterations),
numberOfPoints(0),
costs(std::numeric_limits<double>::max()),
projections(false)
{
    centers = new double*[k];
    finalcenters = new double*[k];
    for (int i = 0; i < k; i++)
    {
        centers[i] = new double[dimension];
        finalcenters[i] = new double[dimension];
    }
    readFile(input);
    readInitialCenters(initcenter);
    /*
          std::cout << "input read" <<std::endl;
          for(int i = 0; i < numberOfPoints; i++) {
                  std::cout<< points[i]->first << " ";
                  for(int j = 0; j < dimension; j++) {
                          std::cout << points[i]->second[j]<<" ";
                  }
                  std::cout << std::endl;
          }
          std::cout << "initial centers:"<<std::endl;
          for(int i = 0; i < k; i++) {
                  for(int j = 0; j < dimension; j++) {
                          std::cout<<centers[i][j] << " ";
                  }
                  std::cout<<std::endl;
          }
     */
}

MASTER::MASTER(char*input, int k, int dimension, int iterations, bool projections) :
dimension(dimension),
k(k),
iterations(iterations),
numberOfPoints(0),
costs(std::numeric_limits<double>::max()),
projections(projections)
{
    centers = new double*[k];
    finalcenters = new double*[k];
    for (int i = 0; i < k; i++)
    {
        centers[i] = new double[dimension];
        finalcenters[i] = new double[dimension];
    }
    readFile(input);
    /*
          std::cout << "input read" <<std::endl;
          for(int i = 0; i < numberOfPoints; i++) {
                  std::cout<< points[i]->first << " ";
                  for(int j = 0; j < dimension; j++) {
                          std::cout << points[i]->second[j]<<" ";
                  }
                  std::cout << std::endl;
          }
     */
}

MASTER::MASTER(char*input, int k, int dimension, int iterations) :
dimension(dimension),
k(k),
iterations(iterations),
numberOfPoints(0),
costs(std::numeric_limits<double>::max()),
projections(false)
{
    centers = new double*[k];
    finalcenters = new double*[k];
    for (int i = 0; i < k; i++)
    {
        centers[i] = new double[dimension];
        finalcenters[i] = new double[dimension];
    }
    readFile(input);
    /*
            std::cout << "input read" <<std::endl;
            for(int i = 0; i < numberOfPoints; i++) {
                    std::cout<< points[i]->first << " ";
                    for(int j = 0; j < dimension; j++) {
                            std::cout << points[i]->second[j]<<" ";
                    }
                    std::cout << std::endl;
            }
     */
}

MASTER::~MASTER()
{
    for (int i = 0; i < numberOfPoints; i++)
    {
        delete points[i]->second;
        delete points[i];
    }
    for (int i = 0; i < k; i++)
    {
        delete centers[i];
        delete finalcenters[i];
    }
    delete centers;
    delete finalcenters;
}

double***MASTER::run(size_t repetitions)
{
    KMEANS algo(this, k, dimension, numberOfPoints, projections);

    //compute initial centers
    double temp;
    int itcount;
    bool converged;
    for (int j = 0; j < repetitions; j++)
    {
        algo.initialCenters(points, centers);
        itcount = 0;
        do
        {
            // assign nearest center for every point and
            // recompute weighted centroid
            converged = algo.reGroupPoints(points, centers);
            itcount++;
            //std::cout << "itcount " << itcount << " iterations " << iterations << " converged " << converged << "\n";
        }
        while (!converged && (itcount < iterations || iterations <= 0));
        std::cout << "Run " << j << " out of " << repetitions << " finished" << std::endl;

        temp = calccosts();
        //std::cout << temp<<std::endl;
        if (temp < costs)
        {
            std::cout << "Better solution found, updating centers" << std::endl;
            costs = temp;
            for (int m = 0; m < k; m++)
            {
                for (int n = 0; n < dimension; n++)
                {
                    finalcenters[m][n] = centers[m][n];
                }
            }
        }
    }
    return &finalcenters;
}

bool MASTER::readFile(char*fileName)
{
    std::FILE*input = std::fopen(fileName, "r");
    double val;
    int countD = 0;
    int countP = 0;
    std::fscanf(input, "%d", &numberOfPoints);
    points = new triple<double, double*, int>*[numberOfPoints];
    while (countP < numberOfPoints && !std::feof(input))
    {
        std::fscanf(input, "%lf", &val);
        if (countD == 0)
        {
            points[countP] = new triple<double, double*, int>(val, new double[dimension], -1);
            countD++;
        }
        else if (countD < dimension)
        {
            points[countP]->second[countD - 1] = val;
            countD++;
        }
        else
        {
            points[countP]->second[dimension - 1] = val;
            countD = 0;
            countP++;
        }
    }
    fclose(input);
    std::cout << "Input file read" << std::endl;
    return true;
}

double MASTER::calccosts()
{
    double val = 0;
    for (int i = 0; i < numberOfPoints; i++)
    {
        val += points[i]->first * squaredDistance(points[i]->second, centers[points[i]->third]);
    }
    return val;
}

double MASTER::squaredDistance(double*p1, double*p2)
{
    double distance = 0;
    double temp = 0;
    for (int j = 0; j < dimension; j++)
    {
        temp = p1[j] - p2[j];
        distance += temp*temp;
    }
    return distance;
}

void MASTER::readInitialCenters(char*initcenter)
{
    std::FILE*input = std::fopen(initcenter, "r");
    int countD = 0;
    int countK = 0;
    double val;
    while (countK < k && !std::feof(input))
    {
        std::fscanf(input, "%lf", &val);
        if (countD < dimension - 1)
        {
            centers[countK][countD] = val;
            countD++;
        }
        else
        {
            centers[countK][countD] = val;
            countD = 0;
            countK++;
        }
    }
    fclose(input);
}

#endif /* MASTER_H_ */
