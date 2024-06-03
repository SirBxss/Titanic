import pandas as pd


class DataLoader:

    @staticmethod
    def load_data(file_path):
        """Load the Titanic dataset from a CSV file."""
        return pd.read_csv(file_path)


# if __name__ == "__main__":
#     file_path = 'Data/train/train.csv'
#     df = load_data(file_path)
#     print(df.head())
