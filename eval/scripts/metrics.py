def recall_at_k(predictions, ground_truth, k):
    """
        Computes the recall at k for a given set of predictions and ground truth labels.

        Args:
            predictions (list): A list of predicted labels.
            ground_truth (list): A list of ground truth labels.
            k (int): The number of top predictions to consider.

        Return:
            float: The recall at k value.
    """
    
    if not ground_truth:
        return 0.0

    if k > len(predictions):
        k = len(predictions)

    top_k_predictions = predictions[:k]
    relevant_items = set(ground_truth)
    retrieved_relevant_items = sum(1 for item in top_k_predictions if item in relevant_items)

    if len(relevant_items) == 0:
        return 0.0

    recall = retrieved_relevant_items / len(relevant_items)
    return recall

def precision_at_k(predictions, ground_truth, k):
    """
        Computes the precision at k for a given set of predictions and ground truth labels.

        Args:
            predictions (list): A list of predicted labels.
            ground_truth (list): A list of ground truth labels.
            k (int): The number of top predictions to consider.

        Return:
            float: The precision at k value.
    """
    
    if not predictions:
        return 0.0

    if k > len(predictions):
        k = len(predictions)

    top_k_predictions = predictions[:k]
    relevant_items = set(ground_truth)
    retrieved_relevant_items = sum(1 for item in top_k_predictions if item in relevant_items)

    if k == 0:
        return 0.0

    precision = retrieved_relevant_items / k
    return precision