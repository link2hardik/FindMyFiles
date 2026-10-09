from eval.scripts.eval_app import EvalApp
from backend.filestore import IngestableFile
import json
import os
from eval.scripts.metrics import recall_at_k, precision_at_k

MIN_FILES_IN_RESULT = 5

def process_files(app:EvalApp, data_dir:str):
    """
    processes all the files by storing them in the EvalApp instance
    """
    manifest_path = os.path.join(data_dir,"manifest.json")
    with open(manifest_path, "r") as m:
        data = json.load(m)
        for file in data:
            file_path = os.path.join(data_dir,"documents",file["filename"])
            with open(file_path,"rb") as f:
                app.add_file(IngestableFile(file_obj=f, name=file["filename"]),
                            file["document_id"])

# TODO : make validate_result and compile_result abstract functions.

def validate_result(result): # to be set up according to the vector store specification
    "Check wether result is valid by checking if the desired number of files are there"
    return len(set([chunk_meta["file_id"] for chunk_meta in result["metadatas"][0]])) >= MIN_FILES_IN_RESULT

def complie_result(app:EvalApp, result): # to be set up according to the vector store specification
    "Complile the result into desirable format."

    result_dict = {}

    for metadata, distance in zip(
        result["metadatas"][0],
        result["distances"][0],
        strict = True
    ):
        file_id_runtime = metadata["file_id"]
        document_id = app.file_map[file_id_runtime]

        if document_id not in result_dict:
            result_dict[document_id] = {"chunk_count": 0, "distances": []}
        result_dict[document_id]["chunk_count"] += 1
        result_dict[document_id]["distances"].append(distance)

    return result_dict
    

def get_predictions(app:EvalApp, questions:list[dict], k_fixed = 10, k_min=10, k_max=100, k_interval=5, is_k_fixed = True):
    """
    Get predictions for the given set of questions, also congigure k
    dynamically to get the best number of files in the search result.
    
    """
    results = {}

    if is_k_fixed:
        k = k_fixed
        for q in questions:
            result = app.search(q["question"],k)
            results[q["question_id"]] = (complie_result(app,result),k)

    else:
        for q in questions:
            k  = k_min
            while k <= k_max:
                result = app.search(q["question"],k)
                if validate_result(result):
                    results[q["question_id"]] = (complie_result(app,result),k)
                    break
                k += k_interval
            if q["question_id"] not in results:
                results[q["question_id"]] = (
                    complie_result(app,result),
                    min(k, k_max),
                )
    
    return results


def compute_metrics(predictions, labels_dict):
    """
    Get the metrics report relevant to the the different configs.
    """
    def rank_results(predictions):
        """
        helper function to convert results (of a single question) with distances and chunk_count to ordered
        document_id using several strategies.
        """
        return sorted(predictions.keys(), key=lambda x: min(predictions[x]["distances"]))
    
    results = {}

    for q_id in predictions:
        ranked_predictions = rank_results(predictions[q_id][0])
        k = min(MIN_FILES_IN_RESULT, len(ranked_predictions))
        labels = labels_dict[q_id]

        results[q_id] = {
            "recall" : recall_at_k(ranked_predictions,labels,k),
            "precision" : precision_at_k(ranked_predictions,labels,k)
        }

    return results

    





        

        