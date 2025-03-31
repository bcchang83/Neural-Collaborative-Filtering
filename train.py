from model import NCF
from data_preprocess import run_data_preprocess
import torch.optim 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(model, epoch_index, tb_writer, training_loader, optimizer, loss_function):
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



def train_model(model, training_loader, validation_loader, epochs_num = 50, learning_rate = 0.001, early_stopping_th = 0.1):
    # Define optimizer and loss function
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = model.loss_fn

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/movielens_training_{}'.format(timestamp))
    # epochs_num = 10
    # early_stopping_th = 0.1
    best_vloss = float("inf")

    loss_history = {
        "training_loss": [],
        "validation_loss": [],
        "validation_loss_best": []
        }
    
    for i in range(epochs_num):
        print(f"Training epoch ={i+1}")
        model.train(True)
        avg_loss = train_one_epoch(model, i, writer, training_loader, optim, loss_fn)
            
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vuser, vmovie, vlabels = vdata
                voutputs = model(vuser, vmovie)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        i + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, i + 1)
            torch.save(model.state_dict(), model_path)

        loss_history["training_loss"].append(avg_loss)
        loss_history["validation_loss"].append(avg_vloss)
        loss_history["validation_loss_best"].append(best_vloss)
        # epoch_number += 1
        # early stopping
        if avg_vloss < early_stopping_th:
            break
    return loss_history