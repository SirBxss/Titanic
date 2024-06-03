from src.Data.DataLoading import DataLoader
from src.Data.DataCleaning import DataCleaner
from src.Features.FeatureEngineer import FeatureEngineer
from src.Visualization.Visualizer import Visualizer
from src.Model.ModelTrainer import ModelTrainer
import pandas as pd


def main():
    train_file_path = 'Data/train/train.csv'
    test_file_path = 'Data/test/test.csv'
    train_df = DataLoader.load_data(train_file_path)
    train_df = DataCleaner.clean_data(train_df)
    train_df = FeatureEngineer.create_feature(train_df)
    Visualizer.plot_survival_rate_by_class(train_df)
    Visualizer.plot_age_distribution(train_df)
    Accuracy = ModelTrainer.train_model(train_df)

    print(train_df)

    model, accuracy = ModelTrainer.train_model(train_df)
    print(f'Accuracy: {Accuracy}')

    test_df = DataLoader.load_data(test_file_path)
    test_df = DataCleaner.clean_data(test_df)
    test_df = FeatureEngineer.create_feature(test_df)

    # Predict on test data
    predictions = ModelTrainer.predict(model, test_df)
    output_file = 'predictions.csv'
    ModelTrainer.save_predictions_to_csv(test_df, predictions, output_file)
    print(f"Predictions saved to {output_file}")
    print(predictions)

    # Optionally: Perform EDA
    Visualizer.plot_survival_rate_by_class(train_df)
    Visualizer.plot_age_distribution(train_df)


if __name__ == '__main__':
    main()
