import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = [9, 10, 13, 18, 19]

# Create a normal probability plot
fig, ax = plt.subplots(figsize=(6,4))
sm.qqplot(np.array(data), line='s', ax=ax)

# Add title and axis labels
plt.title('Normal Probability Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.show()
