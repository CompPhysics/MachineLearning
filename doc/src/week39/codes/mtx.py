import numpy as np
# Design matrix
X = np.array([ [1, 0, 0], [1, 0, 1], [1, 1, 0],[1, 1, 1]],dtype=np.float64)
print(f"The X.TX  matrix:{X.T @ X}")
Xinv = np.linalg.pinv(X.T @ X)
print(f"The invers of X.TX  matrix:{Xinv}")

# The XOR gate 
yXOR = np.array( [ 0, 1 ,1, 0])
ThetaXOR  = Xinv @ X.T @ yXOR
print(f"The values of theta for the XOR gate:{ThetaXOR}")
print(f"The linear regression prediction  for the XOR gate:{X @ ThetaXOR}")


# The OR gate 
yOR = np.array( [ 0, 1 ,1, 1])
ThetaOR  = Xinv @ X.T @ yOR
print(f"The values of theta for the OR gate:{ThetaOR}")
print(f"The linear regression prediction  for the OR gate:{X @ ThetaOR}")


# The OR gate 
yAND = np.array( [ 0, 0 ,0, 1])
ThetaAND  = Xinv @ X.T @ yAND
print(f"The values of theta for the AND gate:{ThetaAND}")
print(f"The linear regression prediction  for the AND gate:{X @ ThetaAND}")
