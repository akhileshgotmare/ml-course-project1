import numpy as np
import csv

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]

    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



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
    return calculate_mse(e)
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
    #w = np.linalg.inv( tx.T.dot(tx) + 2*tx.shape[0]*lamb*np.eye(tx.shape[1]) ).dot( tx.T.dot(y) )
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

def compute_classerror(w, tx_tr, tx_te, y_tr, y_te):
    y_tr_pred = predict_labels(w, tx_tr)
    loss_tr = len(np.nonzero(y_tr_pred-y_tr)[0])/len(y_tr)
    
    y_te_pred = predict_labels(w, tx_te)
    loss_te = len(np.nonzero(y_te_pred-y_te)[0])/len(y_te)
    
    return loss_tr, loss_te

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


##### PLOTS #####

# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
