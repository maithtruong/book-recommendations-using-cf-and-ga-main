import numpy as np
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self):
        pass

    def calculate_predicted_rating_aws(self, target_item_index, user_ratings_input, user_ratings):
        average_rating_target_user = self.average_rating(user_ratings_input)
        total_weighted_rating = 0
        total_similarity = 0
        for user_rating in user_ratings:
            if user_rating[target_item_index] != 0:
                pearson = self.pearson_similarity(user_ratings_input, user_rating)
                jaccard = self.jaccard_coefficient(user_ratings_input, user_rating)
                similarity = pearson * jaccard
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

    def return_actual_predicted(self, user_ratings):
        num_users = len(user_ratings)
        num_items = len(user_ratings[0])

        all_user_predicted_ratings = []
        all_user_actual_ratings = []

        for user_index, user_rating in enumerate(user_ratings):
            filtered_user_ratings = user_ratings[:user_index] + user_ratings[user_index + 1:]

            user_predicted_ratings = []
            user_actual_ratings = []
            
            for index in range(num_items):
                if user_rating[index] == 0:
                    current_item_actual_rating = user_rating[index]
                    current_item_predicted_rating = 0
                    user_predicted_ratings.append(current_item_predicted_rating)
                    user_actual_ratings.append(current_item_actual_rating)

                if user_rating[index] != 0:
                    current_item_actual_rating = user_rating[index]
                    current_item_predicted_rating = self.calculate_predicted_rating_aws(index, user_rating, filtered_user_ratings)
                    user_predicted_ratings.append(current_item_predicted_rating)
                    user_actual_ratings.append(current_item_actual_rating)
            
            all_user_predicted_ratings.append(user_predicted_ratings)
            all_user_actual_ratings.append(user_actual_ratings)
        
        return all_user_predicted_ratings, all_user_actual_ratings

    def calculate_mae(self, all_user_predicted_ratings, all_user_actual_ratings):
        total_mae = 0
        num_users = len(all_user_predicted_ratings)
        num_items = len(all_user_predicted_ratings[0])

        for user_index in range(num_users):
            for item_index in range(num_items):
                predicted_rating = all_user_predicted_ratings[user_index][item_index]
                actual_rating = all_user_actual_ratings[user_index][item_index]
                mae = abs(predicted_rating - actual_rating)
                total_mae += mae

        if num_users > 0 and num_items > 0:
            total_mae = total_mae / (num_users * num_items)

        return total_mae

    def calculate_precision(self, actual_ratings, predicted_ratings, threshold=3):
        true_positives = 0
        predicted_positives = 0

        for actual_user_ratings, predicted_user_ratings in zip(actual_ratings, predicted_ratings):
            for actual, predicted in zip(actual_user_ratings, predicted_user_ratings):
                if actual >= threshold and predicted >= threshold:
                    true_positives += 1
                if predicted >= threshold:
                    predicted_positives += 1

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        return precision

    def calculate_recall(self, actual_ratings, predicted_ratings, threshold=3):
        true_positives = 0
        actual_positives = 0

        for actual_user_ratings, predicted_user_ratings in zip(actual_ratings, predicted_ratings):
            for actual, predicted in zip(actual_user_ratings, predicted_user_ratings):
                if actual >= threshold and predicted >= threshold:
                    true_positives += 1
                if actual >= threshold:
                    actual_positives += 1

        recall = true_positives / actual_positives if actual_positives > 0 else 0
        return recall

    def calculate_f1_measure(self, actual_ratings, predicted_ratings, threshold=3):
        precision = self.calculate_precision(actual_ratings, predicted_ratings, threshold)
        recall = self.calculate_recall(actual_ratings, predicted_ratings, threshold)
        f1_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_measure

    def evaluation(self, user_ratings):
        all_user_predicted_ratings, all_user_actual_ratings = self.return_actual_predicted(user_ratings)
        mae = self.calculate_mae(all_user_predicted_ratings, all_user_actual_ratings)
        precision = self.calculate_precision(all_user_actual_ratings, all_user_predicted_ratings, threshold=3)
        recall = self.calculate_recall(all_user_actual_ratings, all_user_predicted_ratings, threshold=3)
        f1_measure = self.calculate_f1_measure(all_user_actual_ratings, all_user_predicted_ratings, threshold=3)

        table_data = [
            ["Metric", "Value"],
            ["MAE", mae],
            ["Precision", precision],
            ["Recall", recall],
            ["F1 Measure", f1_measure]
        ]

        # Create a table using matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', colColours=['#f5f5f5']*2)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)  # Adjust table size if needed

        # Show the plot with the table
        plt.title("Evaluation Results")
        plt.show()
