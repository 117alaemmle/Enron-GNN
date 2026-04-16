import pandas as pd
import matplotlib.pyplot as plt

# Load and count the predictions
counts = pd.read_csv('unlabeled_predictions.csv')['Predicted_Role'].value_counts()

# Create the bar chart
counts.plot(
    kind='bar', 
    title='Predicted Employee Roles Distribution', 
    rot=45, 
    ylabel='Number of Employees'
)

# This single line fixes the cut-off labels!
plt.tight_layout()

# Display the chart
plt.show()