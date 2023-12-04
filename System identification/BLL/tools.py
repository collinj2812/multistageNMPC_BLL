# Credit: Felix Fiedler 2023
# https://github.com/4flixt/2023_Paper_BLL_LML/

import numpy as np
from sklearn.model_selection import train_test_split
import pdb
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl

class Scaler:
    """Simple wrapper for the sklearn ``preprocessing.StandardScaler`` class.
    
    The scaler handles the scaling of the input and output data simultaneously.

    Args:
        X (np.array): Input data of shape (m, n_x)
        Y (np.array): Output data of shape (m, n_y)
    """
    def __init__(self, X=None,Y=None):
        self.flags = {}

        if X is not None:
            self.scaler_x = preprocessing.StandardScaler().fit(X)
            self.n_x = X.shape[1]

        if Y is not None:
            self.scaler_y = preprocessing.StandardScaler().fit(Y)
            self.n_y = Y.shape[1]


    def scale(self, X=None, Y=None):
        out = []
        if X is not None:
            out.append(self.scaler_x.transform(X))
        if Y is not None:
            out.append(self.scaler_y.transform(Y))

        return out

    def unscale(self, X=None, Y=None):
        out = []
        if X is not None:
            out.append(self.scaler_x.inverse_transform(X))
        if Y is not None:
            out.append(self.scaler_y.inverse_transform(Y))
        return out


def DNN_from_architecture(inputs, architecture):
    """Creates a DNN from a list of layer descriptions.
    Each element of the list must be a tuple of the form (layer_type, keyword_arguments).

    **Example:**
    :: 

        model_input = keras.Input(shape=(data[0].shape[1],))

        # Hidden units
        architecture = [
            (keras.layers.Dense, {'units': 30, 'activation': tf.nn.tanh, 'name': '01_dense'}),
            (keras.layers.Dense, {'units': 30, 'activation': tf.nn.tanh, 'name': '02_dense'}),
            (keras.layers.Dense, {'name': 'output', 'units': data[1].shape[1]})
        ]

    Args:
        inputs (keras.Input): Input layer
        architecture (list): List of layer descriptions (see above)

    """
    outputs = [inputs]
    layers = [inputs]

    for layer_type, layer_config in architecture:
        layers.append(layer_type(**layer_config))
        outputs.append(layers[-1](outputs[-1]))

    return layers, outputs


def blkdiag(*args):
    """Creates a block-diagonal matrix from an arbitrary number of numpy matrices.
    The function is called recursively for multiple arguments.
    """
    A = args[0]
    if len(args)==1:
        B = np.zeros((0,0))
    if len(args)==2:
        B = args[1]

        assert isinstance(A, np.ndarray), 'Argument must be a numpy.ndarray.'
        assert A.ndim ==2, 'Matrix must be 2-dimensional'
        assert isinstance(B, np.ndarray), 'Argument must be a numpy.ndarray.'
        assert B.ndim ==2, 'Matrix must be 2-dimensional'

    if len(args)>2:
        B = blkdiag(*args[1:])



    A_L = np.zeros((A.shape[0],B.shape[1]))
    B_R = np.zeros((B.shape[0],A.shape[1]))
    C = np.block([[A,A_L],[B_R, B]])
    return C


def get_data(m, x_range=[0,1], function_type=0, x_type='random', sigma=0, dtype='float32', random_seed=None):
    """ Function to generate dummy data for testing purposes.

    Function type can be a list or a single integer. If it is a list, a function with multiple outputs will be samples (each output referring to a different function type).

    Args:
        m (int): Number of samples
        x_range (list, tuple): Range of the input data (lower and upper bound)
        function_type (int): Type of the function to be evaluated
        x_type (str): Type of the input data. Either 'random' or 'linear'
        dtype (str): Valid numpy data type to be assigned to the output data
        sigma (float, numpy.ndarray): Standard deviation of the noise. If function_type is a list, sigma can be a list of standard deviations for each function type.
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: Tuple containing the input and output data
    """
    if random_seed:
        np.random.seed(random_seed)

    if isinstance(function_type, int):
        function_type = [function_type]
    if isinstance(sigma,(int,float)):
        sigma = [sigma]

    if x_type == 'random':
        x = np.sort(np.random.rand(m).astype(dtype)).reshape(-1,1)
        x *= x_range[1]-x_range[0]
        x += x_range[0]
    elif x_type == 'linear':
        x = np.linspace(x_range[0], x_range[1], m).astype(dtype).reshape(-1,1)

    y_list = []

    for sigma_i ,func_type in zip(sigma,function_type):
        y = _test_function(x, func_type)
        w = np.random.randn(*y.shape)*sigma_i
        y_list.append(y+w)

    y = np.concatenate(y_list, axis=1).astype(dtype)

    return x,y


def _test_function(x, function_type):
    """Function to generate dummy data for testing purposes. This function is called from ``get_data``.
    """
    if function_type == 0:
        y = 5*x + 2*np.sin(5*np.pi*x)
    elif function_type == 1:
        y = x**2*np.sin(4*x*np.pi)
    elif function_type == 2:
        y = 2*(np.round(4*x)%2)
    elif function_type == 3:
        y = np.sin(4*np.pi*x)
    elif function_type == 4:
        y = 0.1*x**3
    elif function_type == 5:
        y = (3*x + 2)
    elif function_type == 6:
        y = (3*x + 2)*(x<10)
        y += (-2*x+52)*(x>=10)
    elif function_type == 7:
        y =  x%2//1
    else:
        raise Exception('function_type {} is not supported'.format(function_type))
    return y


def split(data, test_size=0.2, random_seed=42):
    """
    Splits the data into training and test sets.

    Args:
        data (tuple): Tuple containing the input and output data
    
    """
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    train = (X_train, y_train)
    test = (X_test, y_test)

    return train, test
 


def figure_1d_data(ax, true=None, train=None, test=None, legend=True):
    """
    Create a figure for 1-D Data give true data, train data and test data. 
    (true,train,test) data must be tuples with (x,y) arrays. 
    
    Returns the figure and axis with the plotted data (for potential post-processing)
    """
    
    if true is not None:
        ax.plot(true[0].flatten(),true[1].flatten(),label='true')
    
    if train is not None:
        ax.set_prop_cycle(None)
        ax.plot(train[0].flatten(),train[1].flatten(),'x', label='measured')
    
    if test is not None:
        ax.plot(test[0].flatten(),test[1].flatten(), label='predicted') 
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if legend == True:
        ax.legend()


def print_percent_done(epoch, total, loss=None, val_loss=None, bar_len=50, title='Please wait'):
    '''
    Simple progress bar. Optionally prints the loss and validation loss.

    Args:
        epoch (int): Current epoch
        total (int): Total number of epochs
        loss (float, optional): Current loss
        val_loss (float, optional): Current validation loss
        bar_len (int, optional): Length of the progress bar
    '''
    percent_done = (epoch+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print_msg = f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done'

    if loss is not None:
        print_msg += f' - loss: {loss:.4f}'
    if val_loss is not None:
        print_msg += f' - val_loss: {val_loss:.4f}'

    print(print_msg, end='\r')


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, line_width=1):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality

    Source: https://gist.github.com/craffel/2d727968c3aaebd10359
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4, linewidth=line_width)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', lw=line_width, alpha=0.5)
                ax.add_artist(line)



def plot_alpha_search(result_train, result_test, alpha_opt, alpha_max, **kwargs):
    """Helper function that can be used with the results of the alpha search for BLL and BLR.
    In both cases, the method ``grid_search_alpha`` returns a dictionary that can be passed either as:
    - ``result_train``: the results of the training set
    - ``result_test``: the results of the test set

    The type of scores that are in the dictionary are considered and the plot is adapted accordingly.

    Keyword arguments are passed to ``plt.subplots``. This can be used e.g. to set the figure size.
    
    """
    
    color = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    log_alpha_test = result_test['log_alpha']
    log_alpha_train = result_train['log_alpha']

    scores_test = list(result_test.keys())
    scores_train = list(result_train.keys())

    assert scores_test == scores_train, 'Not the same metrics for train and test'

    scores_test.remove('log_alpha')
    scores_train.remove('log_alpha')

    fig, ax = plt.subplots(len(scores_test), **kwargs)
    ax_twins = []

    for i, score in enumerate(scores_test):
        ax[i].plot(log_alpha_test, result_test[score], color=color[0], label='test')
        ax[i].set_xlabel('$\log(\\alpha)$')
        ax[i].set_ylabel(score + ' (test)', color=color[0])
        ax[i].tick_params(axis='y', color=color[0], labelcolor=color[0])

        ax_twins.append(ax[i].twinx())
        ax_twins[-1].plot(log_alpha_train, result_train[score], color=color[1], label='train')
        ax_twins[-1].set_ylabel(score + ' (train)', color=color[1])
        ax_twins[-1].tick_params(axis='y', color=color[1], labelcolor=color[1])

        ax[i].axvline(alpha_opt, color='k', linestyle='--', label='opt')
        ax[i].axvline(alpha_max, color='k', linestyle='-', label='max')
    
    ax[0].legend()
    ax[0].set_title('Grid search for $\log(\\alpha)$')
    ax[-1].set_xlabel('$\log(\\alpha)$')
    

    return fig, ax