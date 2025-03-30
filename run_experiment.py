from model import NCF
from data_preprocess import run_data_preprocess

if __name__ == '__main__':
    # load data and preprocess
    path_rating = "./ml-1m/ratings.dat"
    path_user = "./ml-1m/users.dat"
    path_movie = "./ml-1m/movies.dat"
    
    train_dataset, test_dataset, val_dataset, num_users, num_movies = run_data_preprocess(path_rating=path_rating, path_user=path_user, path_movie=path_movie)

    # build model
    model = NCF(num_users=num_users, num_movies=num_movies)
    # train & optimize

    # evaluation


