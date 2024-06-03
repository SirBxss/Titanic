from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ModelTrainer:
    @staticmethod
    def train_model(df):
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
        X = df[features]
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy  # Return both the model and accuracy

    @staticmethod
    def predict(model, df):
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
        X = df[features]
        return model.predict(X)

    @staticmethod
    def save_predictions_to_csv(df, predictions, output_file):
        df['Predicted'] = predictions
        df[['PassengerId', 'Name', 'Sex', 'Predicted']].to_csv(output_file, index=False)  # Assuming 'ID' column exists
