import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_survival_rate_by_class(df):
        sns.barplot(x='Pclass', y='Survived', data=df)
        plt.title('Survival Rate by Class')
        plt.show()

    @staticmethod
    def plot_age_distribution(df):
        sns.histplot(df['Age'].dropna(), kde=True)
        plt.title('Age Distribution of Passengers')
        plt.show()