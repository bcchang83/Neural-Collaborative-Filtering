import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
def negative_sampling(df_rating, df_movie, df_user):
    """
    Sampling non-interacted movie as label = 0
    Args:
        df_rating (_type_): rating data
        df_movie (_type_): movie data
    """
    # user_id = df_rating.UserID.unique()
    # movie_id = df_movie.MovieID.unique()
    # df_unseen = pd.DataFrame()
    # for user in user_id:
    #     seen_movies = df_rating.loc[df_rating.UserID == user, "MovieID"].unique()
    #     num_seen = len(seen_movies)
    #     unseen_movies = np.array(list(set(movie_id) - set(seen_movies))).astype(int)
    #     # breakpoint()a
    #     unseen_sampled = np.random.choice(unseen_movies, num_seen)
    #     d = {"UserID" : np.array([user] * num_seen),
    #          "MovieID" : unseen_sampled,
    #          "Rating" : np.zeros(num_seen)}
    #     df = pd.DataFrame(d)
    #     df_unseen = pd.concat([df_unseen, df]) 
    # df_add = pd.concat([df_rating, df_unseen])
    # return df_add
    
    #Optimized code
    all_users = df_user.UserID.unique()
    all_movies = df_movie.MovieID.unique()
    
    exist_rating = set(zip(df_rating['UserID'], df_rating['MovieID']))

    df_pos = df_rating[df_rating.Rating >= 4]
    df_neg = df_rating[df_rating.Rating < 4]

    diff = len(df_pos) - len(df_neg)
    sampled_data = []
    while diff > 0:
        user = np.random.choice(all_users)
        movie = np.random.choice(all_movies)
        pair = (user, movie)
        if pair not in exist_rating:
            sampled_data.append((user, movie, 0))
            exist_rating.add(pair)
            diff -=1

    df_unseen = pd.DataFrame(sampled_data, columns=["UserID", "MovieID", "Rating"])
    return pd.concat([df_rating, df_unseen], ignore_index=True)     
            


    # print(df_rating.head(10))
    # all_movies = set(df_movie.MovieID.unique())
    # #get the seen movie set for all user. apply() is faster
    # user_seen_dict = df_rating.groupby("UserID")["MovieID"].apply(set)
    
    # sampled_data = []

    # for user, seen_movies in user_seen_dict.items():
    #     unseen_movies = list(all_movies - seen_movies)
    #     num_seen = len(seen_movies)
        
    #     if len(unseen_movies) == 0:
    #         continue
    #     # Suggest at least as many as they already saw
    #     num_sample = min(num_seen, len(unseen_movies))
    #     unseen_sampled = np.random.choice(unseen_movies, num_sample, replace=False)
        
    #     sampled_data.extend(zip([user] * num_sample, unseen_sampled, [0] * num_sample))

    # df_unseen = pd.DataFrame(sampled_data, columns=["UserID", "MovieID", "Rating"])
    
    # return pd.concat([df_rating, df_unseen], ignore_index=True)
        
        
# refer: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        
        # self.UserID = dataframe["UserID"]
        # self.MovieID = dataframe["MovieID"]
        # self.Label = dataframe["Label"]
        
        # turn to torch tensor will be better for later calculation (model training...)
        # -1 cuz index start from 0
        self.users = torch.tensor(dataframe["UserID"].values - 1, dtype=torch.int32)
        self.items = torch.tensor(dataframe["MovieID"].values, dtype=torch.int32)
        self.labels = torch.tensor(dataframe["Label"].values, dtype=torch.float32)
    
    def __len__(self):
        # return len(self.Label)
        return len(self.labels)

    def __getitem__(self, idx):
        # iloc is slower
        # UserID = self.UserID.iloc[idx]
        # MovieID = self.MovieID.iloc[idx]
        # Label = self.Label.iloc[idx]
        # return UserID, MovieID, Label
        return self.users[idx], self.items[idx], self.labels[idx]

def run_data_preprocess(path_rating, path_user, path_movie):
    df_rating = pd.read_csv(path_rating, sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="ISO-8859-1")
    df_user = pd.read_csv(path_user, sep="::", engine="python", names=["UserID", "Gender", "Age","Occupation", "Zip-code"], encoding="ISO-8859-1")
    df_movie = pd.read_csv(path_movie, sep="::", engine="python", names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1")

    num_users = len(df_user.UserID.unique())
    num_movies = len(df_movie.MovieID.unique())
    print(f"Number of Users = {num_users}\nNumber of Movies = {num_movies}")
    
    # Fix the movie ID problem
    all_movies = df_movie.MovieID.unique()
    movieID_map = {}
    for i in range(len(df_movie)):
        movieID_map[all_movies[i]] = i

    df_movie["MovieID"] = df_movie.MovieID.replace(movieID_map)
    df_rating["MovieID"] = df_rating.MovieID.replace(movieID_map)
    
    # Sample non-interacted movies as negatives
    df_rating_add = negative_sampling(df_rating, df_movie, df_user)
    # print(df_rating_add.Rating.value_counts())

    # Label
    df_rating_add["Label"] = -1
    df_rating_add.loc[df_rating_add.Rating >= 4,'Label'] = 1
    df_rating_add.loc[df_rating_add.Rating < 4,'Label'] = 0
    # print(df_rating_add['Label'].value_counts())
    # balance the label?

    df_labeled = df_rating_add[df_rating_add.Label.isin([0, 1])].copy().loc[:,["UserID", "MovieID", "Label"]]
    # df_labeled["Label"] = df_labeled["Label"].astype(int)

    # turn df to matrix? no need for NCF.

    # Split dataset
    # should we do user-based split? the description says random
    # randomly sampling may cause data leakage?
    full_dataset = CustomDataset(df_labeled)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.15, 0.15])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Val size: {len(val_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, num_users, num_movies

if __name__ == '__main__':
    # Load data
    path_rating = "./ml-1m/ratings.dat"
    path_user = "./ml-1m/users.dat"
    path_movie = "./ml-1m/movies.dat"
    
    train_dataset, val_dataset, test_dataset, num_users, num_movies = run_data_preprocess(path_rating=path_rating, path_user=path_user, path_movie=path_movie)


# df_rating = pd.read_csv(path_rating, sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="ISO-8859-1")
# df_user = pd.read_csv(path_user, sep="::", engine="python", names=["UserID", "Gender", "Age","Occupation", "Zip-code"], encoding="ISO-8859-1")
# df_movie = pd.read_csv(path_movie, sep="::", engine="python", names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1")

# num_user = len(df_user.UserID.unique())
# num_movie = len(df_movie.MovieID.unique())
# print(f"Number of Users = {num_user}\nNumber of Movies = {num_movie}")

# # Sample non-interacted movies
# df_rating_add = negative_sampling(df_rating, df_movie)

# # Label
# df_rating_add["Label"] = -1
# df_rating_add.loc[df_rating_add.Rating >= 4,'Label'] = 1
# df_rating_add.loc[df_rating_add.Rating == 0,'Label'] = 0
# print(df_rating_add['Label'].value_counts())
# # balance the label?

# df_labeled = df_rating_add[df_rating_add.Label.isin([0, 1])].copy().loc[:,["UserID", "MovieID", "Label"]]
# # df_labeled["Label"] = df_labeled["Label"].astype(int)

# # turn df to matrix? no need for NCF.

# # Split dataset
# # should we do user-based split? the description says random
# # randomly sampling may cause data leakage?
# full_dataset = CustomDataset(df_labeled)
# train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.15, 0.15])
# print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Val size: {len(val_dataset)}")
