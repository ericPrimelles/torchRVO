import pandas as pd
import matplotlib.pyplot as plt

# Read in the csv file
df = pd.read_csv('rewards.csv')
# df = df[df.index < 500]
# Use the index to plot the rewards
plt.plot(df.index, df['reward'])

# Show the plot 
plt.show()