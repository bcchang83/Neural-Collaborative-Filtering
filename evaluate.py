from model import NCF
from data_preprocess import run_data_preprocess
import torch.optim
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def dcg_at_k(relevance, k):
    """
    calculate dcg
    """
    relevance = np.array(relevance)[:k]  # 取前 k 個
    return np.sum(relevance / np.log2(np.arange(1, len(relevance) + 1) + 1))

def evaluate(model, testing_loader):
    prediction_results = {
        "UserID":[],
        "MovieID":[],
        "Label":[],
        "Prediction":[]
    }
    loss_fn = model.loss_fn
    running_t_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        for i, t_data in enumerate(testing_loader):
            t_user, t_movie, t_labels = t_data
            t_outputs = model(t_user, t_movie)
            t_loss = loss_fn(t_outputs, t_labels)
            running_t_loss += t_loss
            prediction_loss_dict["UserID"].append(*t_user)
            prediction_loss_dict["MovieID"].append(*t_movie)
            prediction_loss_dict["Label"].append(*t_labels)
            prediction_loss_dict["Prediction"].append(*t_outputs)

    avg_t_loss = running_t_loss / (i + 1)
    print(f"Testing loss = {avg_t_loss}")
    
    df = pd.DataFrame(prediction_loss_dict)
    df.sort_values("Prediction", ascending=False, inplace=True)
    df_top10 = df.head(10)
    # Compure recall at 10
    # recall_10 = df_top10["Label"].sum()/10 #this is precision
    recall_10 = df_top10["Label"].sum() / df["Label"].sum()
    print(f"Testing recall@10 = {recall_10}")
    # Compute NDCG@10
    k = 10
    relevance_pred = df["Label"].values[:k]
    dcg_k = dcg_at_k(relevance_pred, k)
    ideal_relevance = np.sort(df["Label"].values)[::-1]
    idcg_k = dcg_at_k(ideal_relevance, k)
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
    print(f"Testing NDCG@{k} = {ndcg_k:.4f}")
    
if __name__ == '__main__':

    saved_model = NCF()
    saved_model.load_state_dict(torch.load('./trained_model'))

    # load data and preprocess
    path_rating = "./ml-1m/ratings.dat"
    path_user = "./ml-1m/users.dat"
    path_movie = "./ml-1m/movies.dat"
    
    train_set, val_set, test_set, num_users, num_movies = run_data_preprocess(path_rating=path_rating, path_user=path_user, path_movie=path_movie)
    
    prediction_loss_dict = {
        "UserID":[],
        "MovieID":[],
        "Label":[],
        "Prediction":[]
    }

    # Evaluation on test data
    loss = torch.nn.BCELoss()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)
    running_t_loss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    saved_model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, t_data in enumerate(test_loader):
            t_user, t_movie, t_labels = t_data
            t_outputs = saved_model(t_user, t_movie)
            t_loss = loss(t_outputs, t_labels)
            running_t_loss += t_loss
            prediction_loss_dict["UserID"].append(*t_user)
            prediction_loss_dict["MovieID"].append(*t_movie)
            prediction_loss_dict["Label"].append(*t_labels)
            prediction_loss_dict["Prediction"].append(*t_outputs)


    avg_t_loss = running_t_loss / (i + 1)

    # Compure recall at 10
    df = pd.DataFrame(prediction_loss_dict)
    df.sort_values("Prediction", ascending=False, inplace=True)
    df_top10 = df.head(10)
    recall_10 = df_top10["Label"].sum()/10

    # NDCG10