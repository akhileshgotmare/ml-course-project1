import numpy as np

# Loss functions
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_rmse(e):
    """Calculate the rmse for vector e."""
    return np.sqrt(np.mean(e**2))

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse, rmse or mae.
    """
    e = y - tx.dot(w)
    #return calculate_rmse(e)
    #return calculate_mse(e)
    return calculate_mae(e)

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def compute_classerror(w, tx_tr, tx_te, y_tr, y_te):
    y_tr_pred = predict_labels(w, tx_tr)
    loss_tr = len(np.nonzero(y_tr_pred-y_tr)[0])/len(y_tr)
    
    y_te_pred = predict_labels(w, tx_te)
    loss_te = len(np.nonzero(y_te_pred-y_te)[0])/len(y_te)
    
    return loss_tr, loss_te

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N, D = tx.shape
    loss = compute_loss(y, tx, w)
    e = y - tx.dot(w)
    g = -tx.T.dot(e)/N

    return g, loss

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        g, loss = compute_gradient(y, tx, w)
        w -= gamma*g
        loss = compute_loss(y, tx, w)

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        # TODO : implement stop criteria

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    N, D = tx.shape
    e = y - tx.dot(w)
    g = -tx.T.dot(e)/N

    return g


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    threshold = 1e-7
    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_epochs):
        g= compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma*g
        loss = compute_loss(y, tx, w)
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws


def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    #loss = compute_loss(y, tx, w_star)

    return w

def build_poly(tX, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    poly = np.zeros((tX.shape[0],1+(tX.shape[1]-1)*degree))
    poly[:,0] = np.ones((tX.shape[0],))
    for p in np.arange(1,degree+1):
        poly[:,1+(p-1)*(tX.shape[1]-1):1+p*(tX.shape[1]-1)] = tX[:,1:tX.shape[1]]**p
    
    return poly

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    N = x.shape[0]
    critical_index = int(N*ratio)
    
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    train_x = shuffled_x[0:critical_index-1]
    train_y = shuffled_y[0:critical_index-1]
    test_x = shuffled_x[critical_index:len(shuffled_x)]
    test_y = shuffled_y[critical_index:len(shuffled_y)]
    
    return train_x, train_y, test_x, test_y

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    
    w = np.linalg.solve(tx.T.dot(tx) + 2*tx.shape[0]*lamb*np.eye(tx.shape[1]), tx.T.dot(y))
    return w


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def predict_logic_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred
            
def compute_logic_classerror(w, tx_tr, tx_te, y_tr, y_te):
    y_tr_pred = predict_logic_labels(w, tx_tr)
    loss_tr = len(np.nonzero(y_tr_pred-y_tr)[0])/len(y_tr)
    
    y_te_pred = predict_logic_labels(w, tx_te)
    loss_te = len(np.nonzero(y_te_pred-y_te)[0])/len(y_te)
    
    return loss_tr, loss_te
            
def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def calculate_logic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    beta = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(beta)) - y*beta)
    return loss

def calculate_logic_gradient(y, tx, w):
    """compute the gradient of loss."""
    g = tx.T.dot(sigmoid(tx.dot(w))-y)
    return g

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_logic_loss(y, tx, w)
    g = calculate_logic_gradient(y, tx, w)

    w -= gamma*g
    
    return loss, w

def learning_by_stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_epochs):
        g= calculate_logic_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma*g
        loss = calculate_logic_loss(y, tx, w)
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws

def calculate_logic_hessian(y, tx, w):
    """return the hessian of the loss function."""
    beta = tx.dot(w)
    S = np.eye(len(y))*sigmoid(beta)*(1-sigmoid(beta))
    H = tx.T.dot(S.dot(tx))
    return H

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""

    loss = calculate_logic_loss(y, tx, w)
    g = calculate_logic_gradient(y, tx, w)
    H = calculate_logic_hessian(y, tx, w)
    
    return loss, g, H

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """

    loss, g, H = logistic_regression(y, tx, w)
    w -= gamma*np.linalg.solve(H,g)
    
    return loss, w

def learning_by_stochastic_newton_method(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_epochs):
        loss, w = learning_by_newton_method(minibatch_y, minibatch_tx, w, gamma)
        loss = calculate_logic_loss(y, tx, w)
        #print("Gradient Descent: loss={l}".format(l=loss))

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    beta = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(beta)) - y*beta) + lambda_*w.T.dot(w)
    g = tx.T.dot(sigmoid(beta)-y) + 2*lambda_*w
    S = np.eye(len(y))*sigmoid(beta)*(1-sigmoid(beta)) 
    H = tx.T.dot(S.dot(tx)) + np.eye(len(w))*2*lambda_
    
    return loss, g, H

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, g, H = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    w -= gamma*np.linalg.solve(H,g)
    
    return loss, w

def learning_by_stochastic_penalized_newton_method(y, tx, initial_w, batch_size, max_epochs, gamma, lambda_):
    """Stochastic gradient descent algorithm."""
    ws = []
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_epochs):
        loss, w = learning_by_penalized_gradient(minibatch_y, minibatch_tx, w, gamma, lambda_)
        loss = calculate_logic_loss(y, tx, w)
        #print("Gradient Descent: loss={l}".format(l=loss))

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws


def penalized_logistic_regression2(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    beta = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(beta)) - y*beta) + lambda_*w.T.dot(w)
    g = tx.T.dot(sigmoid(beta)-y) + 2*lambda_*w
    
    return loss, g

def learning_by_penalized_gradient2(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, g = penalized_logistic_regression2(y, tx, w, lambda_)
    # update w
    w -= gamma*g
    
    return loss, w

def learning_by_stochastic_penalized_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma, lambda_):
    """Stochastic gradient descent algorithm."""
    ws = []
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_epochs):
        loss, w = learning_by_penalized_gradient2(minibatch_y, minibatch_tx, w, gamma, lambda_)
        loss = calculate_logic_loss(y, tx, w)

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws