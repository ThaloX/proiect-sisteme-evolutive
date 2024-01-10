from random import choices, randint, randrange, random, sample
from typing import List, Optional, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


def generate_genome(length: int) -> Genome:
    """
    Generates a random genome of the specified length.

    Parameters:
    length (int): The length of the genome to generate.

    Returns:
    Genome: The randomly generated genome.
    """
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    """
    Generates a population of genomes.

    Args:
        size (int): The number of genomes in the population.
        genome_length (int): The length of each genome.

    Returns:
        Population: A list of generated genomes.
    """
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """
    Performs single-point crossover between two genomes.

    Args:
        a (Genome): The first genome.
        b (Genome): The second genome.

    Returns:
        Tuple[Genome, Genome]: A tuple containing the two offspring genomes resulting from the crossover.
    """
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    """
    Mutates the given genome by randomly flipping bits.

    Args:
        genome (Genome): The genome to be mutated.
        num (int, optional): The number of mutations to perform. Defaults to 1.
        probability (float, optional): The probability of each bit being flipped. Defaults to 0.5.

    Returns:
        Genome: The mutated genome.
    """
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    """
    Calculate the total fitness of a population.

    Args:
        population (Population): The population of genomes.
        fitness_func (FitnessFunc): The fitness function to evaluate each genome.

    Returns:
        int: The total fitness of the population.
    """
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Selects a pair of individuals from the population based on their fitness.

    Args:
        population (Population): The population of individuals.
        fitness_func (FitnessFunc): The fitness function used to evaluate the individuals.

    Returns:
        Population: A pair of selected individuals from the population.
    """
    return sample(
        population=generate_weighted_distribution(population, fitness_func),
        k=2
    )


def generate_weighted_distribution(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Generates a weighted distribution of the population based on the fitness function.

    Args:
        population (Population): The population of genes.
        fitness_func (FitnessFunc): The fitness function used to evaluate the genes.

    Returns:
        Population: The weighted distribution of the population.
    """
    result = []

    for gene in population:
        result += [gene] * int(fitness_func(gene)+1)

    return result


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Sorts the population based on the fitness values calculated using the fitness function.

    Args:
        population (Population): The population to be sorted.
        fitness_func (FitnessFunc): The fitness function used to calculate the fitness values.

    Returns:
        Population: The sorted population.
    """
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    """
    Converts a genome to a string representation.

    Args:
        genome (Genome): The genome to convert.

    Returns:
        str: The string representation of the genome.
    """
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    """
    Prints the statistics of a population for a given generation.

    Args:
        population (Population): The population of genomes.
        generation_id (int): The ID of the current generation.
        fitness_func (FitnessFunc): The fitness function used to evaluate the genomes.

    Returns:
        Genome: The best genome in the population.
    """
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]), fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    """
    Runs the evolution process to generate a population of genomes that meet the fitness limit.

    Args:
        populate_func (PopulateFunc): A function that generates the initial population.
        fitness_func (FitnessFunc): A function that calculates the fitness of a genome.
        fitness_limit (int): The fitness limit that the genomes need to meet.
        selection_func (SelectionFunc, optional): A function that selects parents for crossover. Defaults to selection_pair.
        crossover_func (CrossoverFunc, optional): A function that performs crossover between two parents. Defaults to single_point_crossover.
        mutation_func (MutationFunc, optional): A function that performs mutation on offspring genomes. Defaults to mutation.
        generation_limit (int, optional): The maximum number of generations to run. Defaults to 100.
        printer (Optional[PrinterFunc], optional): A function that prints the population and generation number. Defaults to None.

    Returns:
        Tuple[Population, int]: A tuple containing the final population and the number of generations run.
    """
    population = populate_func()

    i = 0
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i
