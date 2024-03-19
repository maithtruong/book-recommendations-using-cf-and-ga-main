import time
import pandas as pd
import matplotlib.pyplot as plt
from bliga import BookRecommendationSystem

class HyperparameterTuner:
    def __init__(self, user_ratings_input, user_ratings, semantic):
        self.user_ratings_input = user_ratings_input
        self.user_ratings = user_ratings
        self.semantic = semantic

    def run_tuning(self, population_size_range, mutation_rate_range, num_generations_range, crossover_funcs, mutation_funcs, no_rec_range):
        results = []
        for population_size in population_size_range:
            for mutation_rate in mutation_rate_range:
                for num_generations in num_generations_range:
                    for crossover_func in crossover_funcs:
                        for mutation_func in mutation_funcs:
                            for no_rec in no_rec_range:
                                try:
                                    start_time = time.time()

                                    recommender = BookRecommendationSystem(self.user_ratings_input, self.user_ratings, self.semantic)
                                    best_solution, semratings_dict, fitness_scores_dict, best_solutions_by_generation = recommender.genetic_algorithm(population_size, mutation_rate, num_generations, crossover_func, mutation_func, no_rec)
                                    elapsed_time = time.time() - start_time

                                    best_total_rating = max(fitness_scores_dict[solution]["total_predicted_rating"] for solution in fitness_scores_dict)
                                    best_semrating = max(semratings_dict[solution] for solution in semratings_dict)
                                    best_solution_fitness = fitness_scores_dict[best_solution]["total_predicted_rating"]
                                    best_solution_semrating = semratings_dict[best_solution]

                                    results.append((population_size, mutation_rate, num_generations, crossover_func, mutation_func, no_rec, best_total_rating, best_semrating, best_solution, best_solution_fitness, best_solution_semrating, elapsed_time))

                                except Exception as e:
                                    print(f"Error occurred for population_size={population_size}, mutation_rate={mutation_rate}, num_generations={num_generations}, crossover_func={crossover_func}, mutation_func={mutation_func}, no_rec={no_rec}: {e}")

        results = pd.DataFrame(results, columns=["population_size", "mutation_rate", "num_generations", "crossover_func", "mutation_func", "no_rec", "best_total_rating", "best_semrating", "best_solution", "best_solution_fitness", "best_solution_semrating", "elapsed_time"])
        results.sort_values(by=["elapsed_time"], ascending=True, inplace=True)
        return results

    def plot_graphs(self):
        # Plotting time vs. best total rating
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results["elapsed_time"], self.results["best_total_rating"], alpha=0.6)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Best Total Rating')
        plt.title('Time vs. Best Total Rating')
        plt.grid(True)
        plt.savefig('time_vs_best_total_rating.png')
        plt.show()

        # Plotting best semrating vs. best total rating
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results["best_semrating"], self.results["best_total_rating"], alpha=0.6)
        plt.xlabel('Best Semrating')
        plt.ylabel('Best Total Rating')
        plt.title('Best Semrating vs. Best Total Rating')
        plt.grid(True)
        plt.savefig('best_semrating_vs_best_total_rating.png')
        plt.show()

        # Plotting mutation rate vs. best total rating
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results["mutation_rate"], self.results["best_total_rating"], alpha=0.6)
        plt.xlabel('Mutation Rate')
        plt.ylabel('Best Total Rating')
        plt.title('Mutation Rate vs. Best Total Rating')
        plt.grid(True)
        plt.savefig('mutation_rate_vs_best_total_rating.png')
        plt.show()

        # Plotting population size vs. best total rating
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results["population_size"], self.results["best_total_rating"], alpha=0.6)
        plt.xlabel('Population Size')
        plt.ylabel('Best Total Rating')
        plt.title('Population Size vs. Best Total Rating')
        plt.grid(True)
        plt.savefig('population_size_vs_best_total_rating.png')
        plt.show()