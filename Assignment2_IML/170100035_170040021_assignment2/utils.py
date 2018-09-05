import numpy as np
 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  return np.sum((1-func[np.where(func<1)])**2)

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  return  np.sum(np.log(1+np.exp(-func)))

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  return -np.sum(func[np.where(func<0)])

def L2_regulariser(weights):
    # Write the L2 loss here
    return np.sum(np.power(weights[:-1],2))

def L4_regulariser(weights):
    # Write the L4 loss here
    return np.sum(np.power(weights[:-1], 4))

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  grad=np.zeros(inputs.shape[1])
  for i in range(len(targets)):
      if func[i]<1:
          grad[:]=grad[:] - 2*targets[i]*inputs[i][:]*(1-func[i])
  return grad

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  grad=np.zeros(inputs.shape[1])
  for i in range(len(targets)):
      grad[:]=grad[:] - (targets[i]*inputs[i][:]*np.exp(-func[i]))/(1+np.exp(-func[i]))
  return grad

def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  targets[targets==0]=-1
  func=np.multiply(targets, outputs)
  grad=np.zeros(inputs.shape[1])
  for i in range(len(targets)):
      if func[i]<0:
          grad[:]=grad[:] - targets[i]*inputs[i][:]
  return grad

def L2_grad(weights):
    # Write the L2 loss gradient here
    return np.append(2*np.abs(weights[:-1]), weights[-1])

def L4_grad(weights):
    # Write the L4 loss gradient here
    return np.append(4*np.abs(np.power(weights[:-1],3)), weights[-1])

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
