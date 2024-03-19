import csv
import random

def import_user_ratings(file_name):
    file_path = f"dataset/{file_name}"
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        user_ratings = [[int(value) for value in row] for row in reader]
    return user_ratings

def import_semantic(file_name):
    file_path = f"dataset/{file_name}"
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        semantic = [[int(value) for value in row] for row in reader]
    return semantic

def generate_random_user_ratings_input(size):
    return [random.randint(0, 5) for _ in range(size)]
