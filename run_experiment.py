
import torch.optim 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from train import train_model
from evaluate import evaluate
from model import NCF
from data_preprocess import run_data_preprocess
from visualization import plot_training_history


def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, loss_function):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
    
        # Every data instance is an input + label pair
        # print(data)
        user, movie, label = data


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(user, movie)

        # Compute the loss and its gradients
        loss = loss_function(outputs, label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss), end="\r", flush=True)
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == '__main__':
    # data path
    path_rating = "./ml-1m/ratings.dat"
    path_user = "./ml-1m/users.dat"
    path_movie = "./ml-1m/movies.dat"
    
    # hyperparameter for model
    embedding_dim = 32
    dropout=0.2
    num_mlp_layers=3
    GMF=True
    MLP=True
    
    # hyperparameter for training
    epochs_num = 10
    batch_size = 256
    learning_rate = 0.001
    early_stopping_th = 0.1 # earlystopping threshold
    weight_decay = 1e-5 # l2 regularization
    
    # load data and preprocess
    train_set, val_set, test_set, num_users, num_movies = run_data_preprocess(path_rating=path_rating, path_user=path_user, path_movie=path_movie)
    
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
    
    # Here model is trained and saved, time to test!
    evaluate(model, testing_loader, save_path=save_path)
    
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

    



