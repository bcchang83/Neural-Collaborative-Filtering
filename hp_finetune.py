import itertools
import torch.optim
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from train import train_model
from evaluate import evaluate
from model import NCF
from data_preprocess import run_data_preprocess
from visualization import plot_training_history

def run_experiment(edim, dropout, num_mlp_layers, GMF, MLP, train_set, val_set, test_set, num_users, num_movies):
    
    # hyperparameter for model
    embedding_dim = edim
    dropout=dropout
    num_mlp_layers=num_mlp_layers
    GMF=GMF
    MLP=MLP
    # hyperparameter for training
    epochs_num = 100
    batch_size = 256
    learning_rate = 0.001
    early_stopping_th = 0.1 # earlystopping threshold
    weight_decay = 1e-5 # l2 regularization
    
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testing_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # build model
    model = NCF(num_users=num_users,
                num_movies=num_movies,
                embedding_dim=embedding_dim,
                dropout=dropout,
                num_mlp_layers=num_mlp_layers,
                GMF=GMF, MLP=MLP)

    print("Model Structure")
    print(model)
    print('='*80)
    # train model
    if GMF and MLP:
        save_path = f"model_edim{embedding_dim}_drop{dropout}_numlayer{num_mlp_layers}_NCF"
    elif GMF:
        save_path = f"model_edim{embedding_dim}_drop{dropout}_numlayer{num_mlp_layers}_GMF"
    elif MLP:
        save_path = f"model_edim{embedding_dim}_drop{dropout}_numlayer{num_mlp_layers}_MLP"
    else:
        raise
    
    training_history = train_model(model=model,
                            training_loader=training_loader,
                            validation_loader=validation_loader,
                            epochs_num=epochs_num,
                            learning_rate=learning_rate,
                            early_stopping_th=early_stopping_th,
                            weight_decay=weight_decay,
                            save_path=save_path)
    
    plot_training_history(training_history, save_path=save_path)
    # should load the best weight here
    model_file = [f for f in os.listdir(save_path) if "model" in f]
    if len(model_file) > 1:
        print("Warning: find one more model")
    elif len(model_file) == 0:
        print("Warning: don't find the best model. use the model from training")
    else:
        print("Load the best model for testing")
        
    model.load_state_dict(torch.load(os.path.join(save_path, model_file[0])))
    evaluate(model, testing_loader, save_path=save_path)
    
if __name__ == '__main__':
    
    # data path
    path_rating = "./ml-1m/ratings.dat"
    path_user = "./ml-1m/users.dat"
    path_movie = "./ml-1m/movies.dat"
    
    # load data and preprocess
    train_set, val_set, test_set, num_users, num_movies = run_data_preprocess(path_rating=path_rating, path_user=path_user, path_movie=path_movie)
    
    hyperparameter_space = {
    "edim": [8, 16, 32],
    "dropout": [0, 0.2],
    "num_mlp_layers": [1, 3],
    "GMF": [True, False],
    "MLP": [True, False]
    }
    

    all_combinations = list(itertools.product(
        hyperparameter_space["edim"],
        hyperparameter_space["dropout"],
        hyperparameter_space["num_mlp_layers"],
        hyperparameter_space["GMF"],
        hyperparameter_space["MLP"]
    ))

    param_grid = [
        {
            "edim": edim,
            "dropout": dropout,
            "num_mlp_layers": num_mlp_layers,
            "GMF": gmf,
            "MLP": mlp
        }
        for edim, dropout, num_mlp_layers, gmf, mlp in all_combinations 
        if (gmf or mlp)]
    
    print(f"all combination = {len(param_grid)}")


    for i in range(len(param_grid)):
        print(f"hyperparameter tuning {i+1} / {len(param_grid)}")
        edim = param_grid[i]["edim"]
        dropout = param_grid[i]["dropout"]
        num_mlp_layers = param_grid[i]["num_mlp_layers"]
        GMF = param_grid[i]["GMF"]
        MLP = param_grid[i]["MLP"]
        
        print(f"edim = {edim}, dropout = {dropout}, num_mlp_layers = {num_mlp_layers}, GMF = {GMF}, MLP = {MLP}")
        
        run_experiment(edim=edim, dropout=dropout, num_mlp_layers=num_mlp_layers, GMF=GMF, MLP=MLP,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        num_users=num_users, num_movies=num_movies)
    
    
    df = pd.DataFrame()
    result_path = [path for path in os.listdir() if "model_edim" in path]
    for path in result_path:
        csv_path = os.path.join(path, "testing_metrics.csv")
        tmp = pd.read_csv(csv_path)
        tmp["model"] = path
        df = pd.concat([df, tmp])
    df.to_csv("hp_finetune_results.csv")
    
    
    # Here model is trained and saved, time to test!
    
    
    # # Define optimizer and loss function
    # optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss = torch.nn.BCELoss()

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/movielens_training_{}'.format(timestamp))
    # epochs_num = 10
    # early_stopping_th = 0.1
    # best_vloss = float("-inf")

    # for i in range(epochs_num):
    #     print(f"Training epoch ={i+1}")
    #     model.train(True)
    #     avg_loss = train_one_epoch(i, writer, training_loader, optim, loss)
            
    #     running_vloss = 0.0
    #     # Set the model to evaluation mode, disabling dropout and using population
    #     # statistics for batch normalization.
    #     model.eval()

    #     # Disable gradient computation and reduce memory consumption.
    #     with torch.no_grad():
    #         for i, vdata in enumerate(validation_loader):
    #             vuser, vmovie, vlabels = vdata
    #             voutputs = model(vuser, vmovie)
    #             vloss = loss(voutputs, vlabels)
    #             running_vloss += vloss

    #     avg_vloss = running_vloss / (i + 1)
    #     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    #     # Log the running loss averaged per batch
    #     # for both training and validation
    #     writer.add_scalars('Training vs. Validation Loss',
    #                     { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                     i + 1)
    #     writer.flush()

    #     # Track best performance, and save the model's state
    #     if avg_vloss < best_vloss:
    #         best_vloss = avg_vloss
    #         model_path = 'model_{}_{}'.format(timestamp, i + 1)
    #         torch.save(model.state_dict(), model_path)

    #     # epoch_number += 1
    #     # early stopping
    #     if avg_vloss < early_stopping_th:
    #         break

    



