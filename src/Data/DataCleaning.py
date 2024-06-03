class DataCleaner:
    @staticmethod
    def handling_missing_data(df):
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df.drop(columns=['Cabin'], inplace=True)
        return df

    @staticmethod
    def convert_categorical_to_numeric(df):
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        return df

    @staticmethod
    def clean_data(df):
        df = DataCleaner.handling_missing_data(df)
        df = DataCleaner.convert_categorical_to_numeric(df)
        return df

