import numpy as np


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
    # return calculate_mse(e)
    return calculate_rmse(e)
    # return calculate_mae(e)

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    for i in range(w0.shape[0]):
        for j in range(w1.shape[0]):
            w = np.array([w0[i], w1[j]])
            losses[i][j]=compute_loss(y, tx, w)
    # ***************************************************
    return losses

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    N, D = tx.shape
    loss = compute_loss(y, tx, w)
    e = y - tx.dot(w)
    g = -tx.T.dot(e)/N
    # ***************************************************
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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        # TODO : implement stop criteria
        
    print("Gradient Descent : loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    N, D = tx.shape
    e = y - tx.dot(w)
    g = -tx.T.dot(e)/N

    return g


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
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
        print("Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(
              l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w_star = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w_star)
    
    #print("Least Squares: loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w_star[0], w1=w_star[1]))
    return loss, w_star

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    tx = np.zeros((x.shape[0],degree+1))
    for i in range(degree+1):
        tx[:,i] = x**i
    
    return tx

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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N,D = tx.shape
    w = np.linalg.inv(tx.T.dot(tx) + lamb*np.ones(D)).dot(tx.T.dot(y))
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

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    x_tr = x[np.delete(k_indices, (k), axis=0).flatten()]
    y_tr = y[np.delete(k_indices, (k), axis=0).flatten()]
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    tx_tr = my.build_poly(x_tr, degree)
    tx_te = my.build_poly(x_te, degree)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w = my.ridge_regression(y_tr, tx_tr, lambda_)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_tr = my.compute_loss(y_tr, tx_tr, w)
    loss_te = my.compute_loss(y_te, tx_te, w)
    
    return loss_tr, loss_te