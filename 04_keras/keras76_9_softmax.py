import numpy as np
import matplotlib.pyplot as plt

x= np.arange(1,5) # -5~5range, 0.1 간격

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x))  # 큰값은 더 크게 작은값은 더 작게

# softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

y= softmax(x)   

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
plt.title("Softmax Function")
plt.xlabel("x")
plt.ylabel("Softmax(x)")
plt.grid()
plt.show()

