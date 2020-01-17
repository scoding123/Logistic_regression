import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    y[y==0] =-1

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        i=0
        
        while i<max_iterations:
            X_T_w=np.dot(w,X.T)
            with_bias=(X_T_w+b) * y
            delta = (np.where( with_bias <= 0,1,0))*y
            delta_X = np.dot(delta,X)
            delta_X_N = delta_X/N
            w_val_add = step_size * delta_X_N
            w = w+ w_val_add
            b_val_add = step_size*np.sum(delta)/N
            b = b+ b_val_add
            i = i+1
            
          
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        i = 0
        
        while i < max_iterations:
            X_T= X.T
            step_size_val = (step_size/N)
            prod_W_X = np.dot(w,X_T)
            prod_bias= prod_W_X + b
            sig_val = sigmoid(-y*(prod_bias))
            delta =sig_val *y
            delta_X = np.dot(delta,X)
            
            w_val_add =step_size_val *delta_X
            w += w_val_add
            sum_delta =np.sum(delta)
            b_val_add = step_size_val*sum_delta
            b += b_val_add
            i = i+1
         
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    
    exp_val = 1 + np.exp(-z)
    val = np.power(exp_val, -1)
    value = val
    ############################################
    
    
    return value



def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = np.zeros(N)
        for i in range(N):
            prod_W_X = np.dot(w,X.T)+b
            if (prod_W_X[i]) <= 0:
                preds[i] = 0
            else:
                preds[i] = 1
        
        #preds = np.where((np.dot(w, X.T)+b) <= 0, 0, 1)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        #preds = np.zeros(N)
        #preds = np.where(sigmoid(np.dot(w, X.T) + b) >= 0.5, 1, 0)
        for i in range(N):
            prod_W_X = sigmoid(np.dot(w,X.T)+b)
            if (prod_W_X[i]) < 0.5:
                preds[i] = 0
            else:
                preds[i] = 1
        
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds




def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
       

        w = np.column_stack((w, b))
        X = np.column_stack((X,np.ones(N)))
        i=0
        
        while i < max_iterations:
           
            random_index = np.random.randint(N, size=1)
            x,yn = X[random_index,:],y[random_index]
            
            #c*1 matrix --- x_w_T
            x_w_T = np.dot(w, x.T) 
           
            e = np.exp(x_w_T - np.max(x_w_T))
            softmax_val= e / np.sum(e, axis=0, keepdims=True)
            softmax_val[yn] =softmax_val[yn]-  1
            #prod is C*D+1 matrix
            prod = np.dot(softmax_val,x)
            w  -= (step_size*prod)
            i = i+1
        b,w = w[:,D], w[:,:D]
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        
      
        w = np.column_stack((w, b))
        
        x = np.column_stack((X,np.ones(N)))

        

        i = 0
        while i < max_iterations:
            step_val = (step_size/N)
            x_w_T = np.dot(w, x.T) 
            e = np.exp(x_w_T - np.max(x_w_T))
            softmax_val= e / np.sum(e, axis=0, keepdims=True)
            val = softmax_val - np.eye(C)[y].T
            x_val_dot = np.dot(val,x)
            w -=  (step_val*x_val_dot)
            i += 1
        

        b,w = w[:,D], w[:,:D]
       
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
   
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    
    X_W_dot_prod = X.dot(w.T)  
    preds = np.argmax(X_W_dot_prod + b, axis=1)

    assert preds.shape == (N,)
    return preds




        