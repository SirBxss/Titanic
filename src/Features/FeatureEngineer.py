class FeatureEngineer:
    @staticmethod
    def create_feature(df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 1
        df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

        return df