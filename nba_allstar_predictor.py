# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Load  dataset
file_path = '/Users/trevorlovet/Downloads/three_seasons_nba_totals.csv'
nba_data = pd.read_csv(file_path)


# Getting Features and Labels
relevant_columns = ['PTS', 'AST', 'TRB', 'allstar']
nba_relevant_data = nba_data[relevant_columns]

# Dropping Nulls
nba_relevant_data.dropna(inplace=True)

# Split the data into features and target
X = nba_relevant_data[['PTS', 'AST', 'TRB']]
y = nba_relevant_data['allstar']

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Visualize data distribution
sns.pairplot(nba_relevant_data, hue='allstar', markers=["o", "s"], diag_kind="kde")
plt.suptitle("Feature Distribution by All-Star Status", y=1.02)
plt.show()


# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# Predict Player is Allstar
def predict_allstar(name, points, rebounds, assists):
    prediction = log_reg.predict([[points, rebounds, assists]])
    result = "All-Star" if prediction[0] == 1 else "Not All-Star"
    print(f"{name} is predicted to be: {result}")


# Prompt the user for player information
def gather_input():
    player_name = input("Enter player name: ")

    while True:
        try:
            player_points = float(input("Enter points: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for points.")

    while True:
        try:
            player_rebounds = float(input("Enter rebounds: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for rebounds.")

    while True:
        try:
            player_assists = float(input("Enter assists: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number for assists.")

    return player_name, player_points, player_rebounds, player_assists


# Main loop to allow the user to try again
while True:
    predict_allstar(*gather_input())

    try_again = input("Do you want to try again? (yes/no): ").strip().lower()

    if try_again != 'yes':
        break
