#include<iostream>
#include<fstream>
#include<ctime>
#include<cstdlib>
#include<cmath>
using namespace std;
struct Coordinate
{
	double x;
	double y;
};
struct Center
{
	double x;
	double y;
	int sum;
};
struct Chromosome
{
	int permutation[10000];
	double fsw;
	double fitness;
	double p;
};

const double Pm = 0.05;//Mutation Probability
const int POPULATION_SIZE = 100;//Population size
const int MAX_GEN = 200;//Maximum number of generation
const int COORDINATES_SIZE = 10000;
const double INF = 1000000000;
const double c = 2;
const double cm = 1.5;

Coordinate coordinates[COORDINATES_SIZE];
Center centers[POPULATION_SIZE][150];
Center best_centers[150];
Chromosome population[POPULATION_SIZE];
Chromosome best_solution;
int k;
int n;
string filePath;
string outFile;
string centersFile;

void init();
void Genetic_KMeans();
void Initialize_population();
void Calculate_fitness();
void Selection();
void Crossover();
void Mutation();
void KMeans();

void read_coordinates();
void init_centers();
void print_coordinates();
void deal_with_illegal_strings();

void calculate_new_centers();
double calculate_sw(Coordinate a, Center b);
void copy(Chromosome& dest, Chromosome& src);
void crossover_copy(int begin, int end, Chromosome* dest, Chromosome& src);

int main(int argc, char *argv[])
{
    // Should be called with ./gka filePath outFile centerFile numClusters numCoordinates
    filePath = argv[1];
	outFile = argv[2];
	centersFile = argv[3];
    k = atoi(argv[4]);
    n = atoi(argv[5]);
	init();
	Genetic_KMeans();
	print_coordinates();
	return 0;
}
/****Read the coordinates; Init k****/
void init()
{
	srand((unsigned)time(0));
	cout << "Genetic K-Means Algorithm: Init()" << endl;
	read_coordinates();
	init_centers();
}
void read_coordinates()
{
	ifstream infile(filePath);
	char coma;
	for(int i = 0; i < n; i++)
	{
		infile >> coordinates[i].x >> coma >> coordinates[i].y;
		//infile >> coordinates[i].x >> coordinates[i].y;
		if(coordinates[i].x == 0 && coordinates[i].y == 0)
			i--;
	}
}
void init_centers()
{
	int rand_index;
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		rand_index = rand() % n;
		for(int j = 0; j < k; j++)
		{	
			centers[i][j].x = coordinates[rand_index].x;
			centers[i][j].y = coordinates[rand_index].y;
		}	
	}
}
void Genetic_KMeans()
{
	// cout << "/****Genetic KMeans Begin****/" << endl; 
	Initialize_population();
	int geno = MAX_GEN;
	while(geno > 0)
	{
		cout << "geno = " << geno << endl;
		Calculate_fitness();
		Selection();
		Crossover();
		Mutation();
		KMeans();
		geno--;
	}
	cout << fixed << "Best solution: " << -best_solution.fsw << endl;
	// cout << "/****Genetic KMeans End****/" << endl; 
}

void Initialize_population()
{
	//Coding: each allele in the chromosome to take values fom 1-K
	//The initial population is selected randomly
	int per = n / k;//the greatest integer which is less than n / k
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		for(int j = 0; j < n; j++)
			population[i].permutation[j] = -1;
		for(int j = 0; j < k; j++)
		{
			//randomly chosen data points to each cluster j
			for(int l = 0; l < per; l++)
			{
				int rand_index = rand() % n;
				while(population[i].permutation[rand_index] != -1)
				{
					rand_index = rand() % n;
				}
				population[i].permutation[rand_index] = j;
			}
		}
		for(int j = 0; j < n; j++)
		{
			if(population[i].permutation[j] == -1)
			{
				//the rest of the points to randomly chosen clusters
				population[i].permutation[j] = rand() % k;
			}
		}
	}
	best_solution.fsw = -INF;
}

void calculate_new_centers(int pop_index)
{
	//cout << "Calculate_new_centers()" << endl;
	Center* new_centers = new Center[k];
	for(int i = 0; i < k; i++)
	{
		new_centers[i].x = 0;
		new_centers[i].y = 0;
		new_centers[i].sum = 0;
	}
	for(int i = 0; i < n; i++)
	{
		if(!(population[pop_index].permutation[i] >= 0 && population[pop_index].permutation[i] < k))
		{
			cout << "Error: Population " << pop_index <<  "'s " << "permutation[" << i << "] (" << "Center Index Out of Range.)" << endl;
		}
		new_centers[population[pop_index].permutation[i]].x += coordinates[i].x;
		new_centers[population[pop_index].permutation[i]].y += coordinates[i].y;
		new_centers[population[pop_index].permutation[i]].sum = new_centers[population[pop_index].permutation[i]].sum + 1;
	}
	for(int i = 0; i < k; i++)
	{
		if(new_centers[i].sum == 0)
		{
			// cout << "Illegal Strings (TO BE DEAL WITH)" << endl;
			new_centers[i].x = 0;
			new_centers[i].y = 0;
		} 
		else 
		{
			new_centers[i].x = new_centers[i].x / new_centers[i].sum;
			new_centers[i].y = new_centers[i].y / new_centers[i].sum;	
		}
	}
	for(int i = 0; i < k; i++)
	{
		centers[pop_index][i].x = new_centers[i].x;
		centers[pop_index][i].y = new_centers[i].y;
		centers[pop_index][i].sum = new_centers[i].sum;
	}
	delete [] new_centers;	
}

void deal_with_illegal_strings()
{	
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		for(int j = 0; j < k; j++)
		{
			if(centers[i][j].sum == 0)
			{
				//convert illegal strings to legal strings
				int rand_index = rand() % n;
				population[i].permutation[rand_index] = j;
			}
		}
		calculate_new_centers(i);
	}
}

void Calculate_fitness()
{
	//cout << "Calculate_finess()" << endl;
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		calculate_new_centers(i);	
	}
	
	deal_with_illegal_strings();
	
	double fsw_sum = 0;
	double fsw_average = 0;
	double fsw_standard_deviation = 0;
	//the total within-cluster variation SW
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		double SW = 0;
		for(int j = 0; j < n; j++)
		{
			SW += calculate_sw(coordinates[j], centers[i][population[i].permutation[j]]);
		}
		//cout << "SW of population[" << i << "] is " << SW << endl;
		population[i].fsw = -SW;////fsw = -SW
		
		if(population[i].fsw > best_solution.fsw)
		{
			copy(best_solution, population[i]);
			for(int l = 0; l < k; l++)
			{
				best_centers[l].x = centers[i][l].x;
				best_centers[l].y = centers[i][l].y;
			}
		}
		
		fsw_sum += population[i].fsw;
	}	
	fsw_average = fsw_sum / POPULATION_SIZE;
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		//gsw = fsw - (fsw_average - c * fsw_standard_deviation); c is a constant between 1 and 3
		fsw_standard_deviation += (population[i].fsw - fsw_average) * (population[i].fsw - fsw_average);
	}
	fsw_standard_deviation = sqrt(fsw_standard_deviation / POPULATION_SIZE);
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		double gsw = 0;
		gsw = population[i].fsw - (fsw_average - c * fsw_standard_deviation);
		if(gsw >= 0)
			population[i].fitness = gsw;
		else
			population[i].fitness = 0;
		//cout << "fitness of population[" << i << "] is " << population[i].fitness << endl;
	}	
}

void Selection()
{
	double sum_fitness = 0;
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		sum_fitness += population[i].fitness;
	}
	population[0].p = population[0].fitness / sum_fitness;
	for(int i = 1; i < POPULATION_SIZE; i++)
	{
		population[i].p = population[i].fitness / sum_fitness + population[i - 1].p;
	}
	//Roulette wheel strategy
	Chromosome new_population[POPULATION_SIZE];
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		double rand_zero_to_one = rand() % 10000 / 10000.0;
		int j = 0;
		while(population[j].p < rand_zero_to_one)
		{
			j++;
		}
		copy(new_population[i], population[j]);
	}
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		copy(population[i], new_population[i]);
	}
}

void Crossover()
{
	//order crossover
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		//population[i]
		Chromosome* offspring1 = new Chromosome;
		Chromosome* offspring2 = new Chromosome;
		int parent1_index = rand() % POPULATION_SIZE;
		int parent2_index = rand() % POPULATION_SIZE;
		int X = rand() % n;
		int Y = rand() % n;
		if(X > Y)
		{
			int temp = X;
			X = Y;
			Y = temp;
		}
		crossover_copy(X, Y, offspring1, population[parent1_index]);
		crossover_copy(X, Y, offspring2, population[parent2_index]);
		crossover_copy(0, X, offspring1, population[parent2_index]);
		crossover_copy(0, X, offspring2, population[parent1_index]);
		crossover_copy(Y, n, offspring1, population[parent2_index]);
		crossover_copy(Y, n, offspring2, population[parent1_index]);
		for(int j = 0; j < n; j++)
		{
			population[parent1_index].permutation[j] = offspring1->permutation[j];
			population[parent2_index].permutation[j] = offspring2->permutation[j];
		}
		delete offspring1;
		delete offspring2; 
	} 
}

void crossover_copy(int begin, int end, Chromosome* dest, Chromosome& src)
{
	for(int j = begin; j < end; j++)
	{
		dest->permutation[j] = src.permutation[j];
	}
}

void Mutation()
{
	//cout << "Mutation()" << endl; 
	//Each allele in a chromosome is mutated with a probability Pm
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		for(int j = 0; j < n; j++)
		{
			double drand = rand() % 10000 / 10000.0;
			if(drand < Pm)
			{
				double dsw[k];
				double dmax = 0;
				for(int l = 0; l < k; l++)
				{
					dsw[l] = calculate_sw(coordinates[j], centers[i][l]);
					if(dsw[l] > dmax)
					{
						dmax = dsw[l];
					}
				}
				double pj[k];
				double formula = 0;
				for(int l = 0; l < k; l++)
				{
					formula += cm * dmax - dsw[l]; 
				}
				pj[0] = (cm * dmax - dsw[0]) / formula;
				for(int l = 1; l < k; l++)
				{
					pj[l] = (cm * dmax - dsw[l]) / formula + pj[l - 1];
				}
				//randomly select a number from {1,2 ... K}
				double rand_number = rand() % 10000 / 10000.0;
				int l = 0;
				while(pj[l] < rand_number)
				{
					l++;
				}
				//the number selected is l
				if(!(l >=0 && l < k))
				{
					cout << rand_number << endl;
				}
				population[i].permutation[j] = l;	
			}
		}
	}
}

void KMeans()
{
	for(int i = 0; i < POPULATION_SIZE; i++)
	{
		calculate_new_centers(i);
		for(int j = 0; j < n; j++)
		{
			double min = INF;
			int min_index = -1;
			for(int l = 0; l < k; l++)
			{
				double dis = calculate_sw(coordinates[j], centers[i][l]);
				if(dis < min)
				{
					min = dis;
					min_index = l;
				}
			}
			population[i].permutation[j] = min_index;
		}
	}
}
	
/****Basic functions****/
double calculate_sw(Coordinate a, Center b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y); 	
}

void copy(Chromosome& dest, Chromosome& src)
{
	for(int i = 0; i < n; i++)
	{
		dest.permutation[i] = src.permutation[i];
		if(!(src.permutation[i] < k && src.permutation[i] >= 0))
		{
			cout << "Error: Src Error (Center Index Out of Range)" << src.permutation[i] << endl;
		}
	}
	dest.fsw = src.fsw;
}

void print_coordinates()
{
	ofstream outfile(outFile);
	ofstream outfile_centers(centersFile);
	for(int i = 0; i < n; i++)
	{
		outfile << coordinates[i].x << " " << coordinates[i].y << " " << best_solution.permutation[i] << endl;
	}
	// cout << "The centers are: " << endl;
	for(int i = 0; i < k; i++)
	{
		// cout << "Center " << i << ": " << best_centers[i].x << " " << best_centers[i].y << " " << endl;
		outfile_centers << best_centers[i].x << " " << best_centers[i].y << endl;
	}
}
