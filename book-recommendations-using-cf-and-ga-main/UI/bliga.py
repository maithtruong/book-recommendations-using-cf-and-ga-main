import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BookRecommendationSystem:

    def __init__(self, user_ratings_input, user_ratings, semantic):
        self.user_ratings_input = user_ratings_input
        self.user_ratings = user_ratings
        self.semantic = semantic

    def calculate_semrating(self, solution):
        semrating = []
        for i, book_i in enumerate(solution):
            for j in range(i + 1, len(solution)):
                f11 = sum(1 for value_i, value_j in zip(self.semantic[book_i], self.semantic[solution[j]]) if value_i == 1 and value_j == 1)
                f10 = sum(1 for value_i, value_j in zip(self.semantic[book_i], self.semantic[solution[j]]) if value_i == 1 and value_j == 0)
                f01 = sum(1 for value_i, value_j in zip(self.semantic[book_i], self.semantic[solution[j]]) if value_i == 0 and value_j == 1)
                rating = f11 / (f10 + f01 + f11)
                semrating.append(rating)
        return sum(semrating)

    def possible_solutions(self, initial_pop_size, no_rec):
        missing_items = [index for index, rating in enumerate(self.user_ratings_input) if rating == 0]
        population = []
        while len(population) < initial_pop_size:
            combination = random.sample(missing_items, no_rec)
            population.append(combination)
        return population

    # Inside the 'generate_initial_population' method
    def generate_initial_population(self, all_recommendations, pop_size, fitness_scores_dict):
        population = []
        # fitness_scores_dict = {}  # Remove this line as it is already passed as an argument

        # Calculate fitness scores and predicted rating list for all solutions and store them in the fitness_scores_dict
        for solution in all_recommendations:
            total_predicted_rating, predicted_rating_list = self.calculate_total_rating(solution)
            semrating = self.calculate_semrating(solution)  # Calculate the semantic rating for the solution
            if semrating > 0:
                # Convert the 'solution' list into a tuple before using it as a key
                solution = tuple(solution)
                fitness_scores_dict[solution] = {
                    "total_predicted_rating": total_predicted_rating,
                    "predicted_rating_list": predicted_rating_list
                }

        # Filter out solutions with semrating <= 0
        filtered_solutions = list(fitness_scores_dict.keys())

        # Calculate the fitness sum for roulette wheel selection
        fitness_sum = sum(fitness_scores_dict[solution]["total_predicted_rating"] for solution in filtered_solutions)
        
        for _ in range(pop_size):
            selected_solution = None
            while selected_solution is None:
                random_number = random.random()
                cumulative_probability = 0
                for solution in filtered_solutions:
                    total_predicted_rating = fitness_scores_dict[solution]["total_predicted_rating"]
                    cumulative_probability += total_predicted_rating / fitness_sum
                    if random_number <= cumulative_probability:
                        selected_solution = solution
                        break

            population.append(selected_solution)

        return population

    def calculate_total_rating(self, solution):
        total_predicted_rating = 0
        predicted_rating_list = []
        for item_index in solution:
            predicted_rating = self.calculate_predicted_rating_aws(item_index)
            total_predicted_rating += predicted_rating
            predicted_rating_list.append(predicted_rating)
        return total_predicted_rating, predicted_rating_list

    def calculate_predicted_rating_aws(self, target_item_index):
        average_rating_target_user = self.average_rating(self.user_ratings_input)
        total_weighted_rating = 0
        total_similarity = 0
        for user_rating in self.user_ratings:
            if user_rating[target_item_index] != 0:
                pearson_similarity = self.pearson_similarity(self.user_ratings_input, user_rating)
                jaccard_coefficient = self.jaccard_coefficient(self.user_ratings_input, user_rating)
                similarity = pearson_similarity * jaccard_coefficient
                average_rating_user_j = self.average_rating(user_rating)
                adjusted_rating = user_rating[target_item_index] - average_rating_user_j
                total_weighted_rating += similarity * adjusted_rating
                total_similarity += abs(similarity)
        if total_similarity != 0:
            predicted_rating = average_rating_target_user + (total_weighted_rating / total_similarity)
        else:
            predicted_rating = average_rating_target_user
        return predicted_rating

    def pearson_similarity(self, a, b):
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        numerator = np.sum((a - a_mean) * (b - b_mean))
        denominator = np.sqrt(np.sum((a - a_mean)**2) * np.sum((b - b_mean)**2))
        return numerator / denominator

    def jaccard_coefficient(self, a, b):
        intersection = len(set(a) & set(b))
        union = len(set(a) | set(b))
        
        if union == 0:
            return 0.0
        else:
            return intersection / union

    def average_rating(self, user_rating):
        if isinstance(user_rating, int):
            return user_rating
        ratings = [rating for rating in user_rating if rating != 0]
        return np.mean(ratings)

    def selection(self, population, fitness_scores, semratings_dict):
        selected_population = []
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1 / len(population)] * len(population)
        else:
            probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        for _ in range(len(population)):
            while True:
                selected_solution = random.choices(population, probabilities)[0]
                semrating = semratings_dict[selected_solution]
                if semrating > 0:
                    break
            selected_population.append(selected_solution)
        return selected_population

    def one_point_crossover(self, parent1, parent2):
        parent1 = list(parent1)
        parent2 = list(parent2)
        crossover_point = random.randint(0, len(parent1))  # Modified: Allow crossover at the beginning or end
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        child1 = self.remove_duplicates(child1)
        child2 = self.remove_duplicates(child2)
        return tuple(child1), tuple(child2)

    def uniform_crossover(self, parent1, parent2):
        parent1 = list(parent1)
        parent2 = list(parent2)
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        child1 = self.remove_duplicates(child1)
        child2 = self.remove_duplicates(child2)
        return tuple(child1), tuple(child2)

    def swap_mutation(self, solution, all_recommendations, mutation_rate):
        mutated_solution = list(solution)
        for i in range(len(solution)):
            if random.random() < mutation_rate:
                possible_items = all_recommendations[i % len(all_recommendations)]
                mutated_solution[i] = random.choice(possible_items)
        mutated_solution = self.remove_duplicates(mutated_solution)
        return tuple(mutated_solution)

    def inversion_mutation(self, solution):
        mutated_solution = list(solution)
        if len(mutated_solution) <= 1:
            return mutated_solution
        start = random.randint(0, len(mutated_solution) - 1)
        end = random.randint(start + 1, len(mutated_solution))
        mutated_solution[start:end] = reversed(mutated_solution[start:end])
        return mutated_solution
    
    @staticmethod
    def remove_duplicates(solution):
        seen = set()
        return [x for x in solution if not (x in seen or seen.add(x))]

    def genetic_algorithm(self, pop_size, mutation_rate, num_generations, crossover_func, mutation_func, no_rec):
        all_recommendations = self.possible_solutions(pop_size * 2, no_rec)

        fitness_scores_dict = {}  # Initialize the fitness scores dictionary
        population = self.generate_initial_population(all_recommendations, pop_size, fitness_scores_dict)

        # Initialize dictionaries to store fitness scores and semantic ratings
        semratings_dict = {tuple(solution): self.calculate_semrating(solution) for solution in population}

        best_solution = None
        best_total_rating = -float('inf')
        best_semrating = -1

        best_solutions_by_generation = {}

        for generation in range(num_generations):
            next_population = []

            # Elitism: Preserve the best solution from the previous generation in the next population
            if best_solution:
                next_population.append(best_solution)

            # Populate the rest of the next_population with new solutions
            while len(next_population) < pop_size - 1:  # Account for the elitism
                # Selection
                selected_population = self.selection(population, [fitness_scores_dict[tuple(solution)]["total_predicted_rating"] for solution in population], semratings_dict)

                parent1, parent2 = random.sample(selected_population, 2)

                if crossover_func == "one_point":
                    child1, child2 = self.one_point_crossover(parent1, parent2)
                elif crossover_func == "uniform":
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                else:
                    raise ValueError("Invalid crossover function: " + crossover_func)

                if mutation_func == "swap":
                    mutated_child1 = self.swap_mutation(child1, all_recommendations, mutation_rate)
                    mutated_child2 = self.swap_mutation(child2, all_recommendations, mutation_rate)
                elif mutation_func == "inversion":
                    mutated_child1 = self.inversion_mutation(child1)
                    mutated_child2 = self.inversion_mutation(child2)
                else:
                    raise ValueError("Invalid mutation function: " + mutation_func)

                # Only add mutated children to next_population if their length is greater than 2 and their semrating is > 0
                if len(mutated_child1) > 2 and self.calculate_semrating(mutated_child1) > 0:
                    next_population.append(mutated_child1)
                if len(mutated_child2) > 2 and len(next_population) < pop_size - 1:  # Ensure we have enough individuals in the population
                    semrating_child2 = self.calculate_semrating(mutated_child2)
                    if semrating_child2 > 0:
                        next_population.append(mutated_child2)

            if next_population:
                # Calculate fitness scores and predicted rating lists only for new solutions
                for solution in next_population:

                    if tuple(solution) not in fitness_scores_dict:
                        total_predicted_rating, predicted_rating_list = self.calculate_total_rating(solution)

                        fitness_scores_dict[tuple(solution)] = {
                            "total_predicted_rating": total_predicted_rating,
                            "predicted_rating_list": predicted_rating_list
                        }

                    if tuple(solution) not in semratings_dict:
                        semrating = self.calculate_semrating(solution)
                        semratings_dict[tuple(solution)] = semrating
            else:
                # Handle the case where next_population is empty (no valid solutions found)
                # You can choose to add some default or random solutions, or handle it in any other way that suits your algorithm.
                pass

            # Find the best solution in the current generation
            best_solution_in_generation = max(
                (solution for solution in next_population),
                key=lambda solution: fitness_scores_dict[tuple(solution)]["total_predicted_rating"]
            )

            # Update the best solution for the overall generations if needed
            if fitness_scores_dict[tuple(best_solution_in_generation)]["total_predicted_rating"] > best_total_rating:
                best_solution = best_solution_in_generation
                best_total_rating = fitness_scores_dict[tuple(best_solution_in_generation)]["total_predicted_rating"]

            # Use the semrating criteria to find the best solution among the ones with the same best total rating
            if len(next_population) > 1:
                same_best_total_rating_solutions = [
                    solution
                    for solution in next_population
                    if len(solution) > 2 and fitness_scores_dict[tuple(solution)]["total_predicted_rating"] == best_total_rating
                ]

                if len(same_best_total_rating_solutions) > 1:
                    best_semrating_among_same_total_rating = 0
                    for solution in same_best_total_rating_solutions:
                        semrating = semratings_dict[tuple(solution)]
                        if semrating > best_semrating_among_same_total_rating:
                            best_semrating = semrating
                            best_solution = solution

            # Store the best solution for this generation
            best_solutions_by_generation[generation] = best_solution

        # Find the best fitness score
        best_fitness_score = max(
            (fitness_scores_dict[tuple(solution)] for solution in fitness_scores_dict if len(solution) > 2),
            key=lambda x: x["total_predicted_rating"]
        )

        # If there are more than one solution with highest fitness, use semantic similarity as a second criteria
        max_fitness_solutions = [
            solution
            for solution in fitness_scores_dict
            if len(solution) > 2 and
            fitness_scores_dict[tuple(solution)]["total_predicted_rating"] == best_fitness_score["total_predicted_rating"]
        ]

        if len(max_fitness_solutions) > 0:
            for solution in max_fitness_solutions:

                # Double check if semantic similarity has not been calculated for solution
                if solution not in semratings_dict:
                    semratings_dict[tuple(solution)] = self.calculate_semrating(solution)
                
                semrating = semratings_dict[tuple(solution)]

                if semrating > best_semrating:
                    best_semrating = semrating
                    best_solution = solution
            else:
                best_solution = max((solution for solution in fitness_scores_dict if len(solution) > 2), key=lambda x: fitness_scores_dict[tuple(x)]["total_predicted_rating"])
            
        # Return the best solution, semratings_dict, fitness_scores_dict, and best_solutions_by_generation
        return best_solution

    def plot_best_average_solutions_by_generation(self, fitness_scores_dict, best_solutions_by_generation):
        generations = list(best_solutions_by_generation.keys())
        
        # Calculate average fitness scores for each generation
        fitness_scores = [fitness_scores_dict[best_solutions_by_generation[gen]]["total_predicted_rating"] / len(best_solutions_by_generation[gen])
                        for gen in generations]

        plt.figure(figsize=(8, 6))
        plt.plot(generations, fitness_scores, marker='o', linestyle='-')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness Score')
        plt.title('Average Fitness Score Evolution')
        plt.grid(True)
        plt.savefig('fitness_graph.png')  # Save the plot as an image
        plt.show()

    def plot_best_total_solutions_by_generation(self, fitness_scores_dict, best_solutions_by_generation):
        generations = list(best_solutions_by_generation.keys())
        
        # Get the total fitness score (total predicted rating) for each generation
        fitness_scores = [
            fitness_scores_dict[solution]["total_predicted_rating"]
            for solution in best_solutions_by_generation.values()
        ]

        plt.figure(figsize=(8, 6))
        plt.plot(generations, fitness_scores, marker='o', linestyle='-')
        plt.xlabel('Generation')
        plt.ylabel('Total Fitness Score')
        plt.title('Total Fitness Score Evolution')
        plt.grid(True)
        plt.savefig('fitness_graph.png')  # Save the plot as an image
        plt.show()

    



