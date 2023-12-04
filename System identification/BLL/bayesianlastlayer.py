# Credit: Felix Fiedler 2023
# Published in "F. Fiedler and S. Lucia (2023). Improved Uncertainty Quantification for Neural Networks With Bayesian Last Layer. IEEE Access, 11, 123149–123160"
# https://github.com/4flixt/2023_Paper_BLL_LML/

from attr import has
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pdb
import tools
import copy
import pickle
from pathlib import Path

class LogMarginalLikelihood:
    """Train :py:class:`bll.BayesianLastLayer` with log marginal likelihood.

    This class is inherited to :py:class:`bll.BayesianLastLayer` and cannot be used on its own.

    Args:
        log_alpha_0 (float, int): Initial value for the log of the signal to noise ratio. 
        log_sigma_e_0 (float, int): Initial value for the log of the noise standard deviation.
    """
    def __init__(self, log_alpha_0 = 4, log_sigma_e_0 = -1):

        self.flags = {
            'setup_training': False,
        }

        # Create vector for initial values of log_sigma_e
        log_sigma_e_0 = np.ones(self.n_y, dtype='float32')*log_sigma_e_0

        self._log_alpha = tf.Variable(name='log_alpha', initial_value=log_alpha_0, dtype='float32')
        self.log_sigma_e = tf.Variable(name='log_sigma_e', initial_value=log_sigma_e_0, dtype='float32')

    @property
    def log_sigma_w(self):
        """Retrieve ``log_sigma_w`` from ``alpha`` and ``log_sigma_e``.

        When setting a value for ``log_sigma_w``, the value for ``alpha`` is updated accordingly.
        """
        return self.log_alpha/2 + self.log_sigma_e

    @log_sigma_w.setter
    def log_sigma_w(self, log_sigma_w):
        print('overwriting log_alpha from log_sigma_w and log_sigma_e')
        self.log_alpha = 2*log_sigma_w - 2*self.log_sigma_e

    @property
    def log_alpha(self):
        """Retrieve ``log_alpha`` or update ``log_alpha``.
        
        When setting a value for ``log_alpha`` and if training was already completed,
        the values of ``Lambda_p_bar`` and ``Sigma_p_bar``, 
        which are used to compute the predictive distribution are updated. 
        """
        return self._log_alpha

    @log_alpha.setter
    def log_alpha(self, log_alpha):
        self._log_alpha = 0*self._log_alpha + log_alpha

        # Update Sigma_bar if prediction has been prepared
        if self.flags['prepare_prediction']:
            self.Lambda_p_bar = self.get_Lambda_p_bar(self.Phi)
            self.Sigma_p_bar = np.linalg.pinv(self.Lambda_p_bar)


    def setup_training(self, optimizer, train_alpha=True, train_sigma_e=True):
        """Setup the training of the model.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to use for training.
            train_alpha (bool): If True, the signal to noise ratio is trained.
            train_sigma_e (bool): If True, the noise standard deviation is trained.
        """

        # Weights and biases of the model are always trained
        self.trainable_variables = self.joint_model.trainable_variables
        # Add log_alpha and log_sigma_e to trainable variables if desired
        if train_alpha:
            self.trainable_variables.append(self.log_alpha)
        if train_sigma_e:
            self.trainable_variables.append(self.log_sigma_e)

        self.optimizer = optimizer
        # Initialize or reset the training history
        self.training_history = {'loss': [], 'val_loss': [], 'epochs': []}

        # Update flags
        self.flags['setup_training'] = True


    def get_Lambda_p_bar(self, Phi):
        """Compute the scaled posterior precision matrix.
        
        Args:
            Phi (tf.Tensor): Feature matrix of the neural network with bias of shape (m, n_phi).
            log_alpha (tf.Tensor, float): Signal to noise ratio.

        Returns:
            Lambda_p_bar (tf.Tensor): Scaled posterior precision matrix of shape (n_phi, n_phi).
        """
        alpha_inv = tf.exp(-self.log_alpha)
        Alpha_inv =  alpha_inv*tf.constant(np.eye(self.n_phi), dtype='float32')

        # Compute Lambda_p_bar
        Lambda_p_bar = Alpha_inv + tf.linalg.matmul(tf.transpose(Phi),Phi)

        return Lambda_p_bar

 
    def lml(self, x, y, training=False):
        """Compute the log marginal likelihood of the model.
        This function is used for training the model.

        Args:
            x (tf.Tensor or numpy.ndarray): Input data of shape (m, n_x).
            y (tf.Tensor or numpy.ndarray): Output data of shape (m, n_y).
        """
        # Number of data points
        m = x.shape[0]

        inv_sigma2_e = tf.exp(-2*self.log_sigma_e)
        inv_sigma2_w = tf.exp(-2*self.log_sigma_w)

        Phi_tilde, y_hat = self.joint_model(x, training=training)
        # Concat vector of ones to Phi_tilde
        Phi = tf.concat([Phi_tilde, tf.ones((m,1))], axis=1)
        
        Lambda_p_bar = self.get_Lambda_p_bar(Phi)

        # Retrieve weights and bias for affine operation in last layer
        w_1 =self.joint_model.layers[-1].weights[0]
        w_2 =tf.reshape(self.joint_model.layers[-1].weights[1],(1,-1))

        # Concatenate weights and bias for equivalent linear operation
        w = tf.concat([w_1, w_2],axis=0)

        
        # Terms used in formulation of cost:
        dy = (y-y_hat)
        dy_square = tf.math.square(dy)
        w_square  = tf.math.square(w)

        # Add the individual terms to the negative log marginal likelihood
        J = 0
        J += self.n_y/(2)   * tf.math.log(2*np.pi)
        J += self.n_y/(2*m) * tf.linalg.logdet(Lambda_p_bar)
        J += self.n_y/(2*m) * (self.n_phi)*self.log_alpha

        
        for i in range(self.n_y):  
            J += self.log_sigma_e[i]
            J += 1/(2*m) * tf.reduce_sum(dy_square[:,i])*inv_sigma2_e[i]
            J += 1/(2*m) * tf.reduce_sum(w_square[:,i]) *inv_sigma2_w[i]

        return J


    def exact_lml(self, x, y):
        """Compute the exact log marginal likelihood of the model.
        Exact means that we compute the weights of the last layer with the explicit least squares solution 
        and obtain the predicted y_hat by mapping the nonlinear features to the output with these weights.

        This method is implemented for testing purposes and not used for training.

        Args:
            x (tf.Tensor or numpy.ndarray): Input data of shape (m, n_x).
            y (tf.Tensor or numpy.ndarray): Output data of shape (m, n_y).
        """
        # Number of data points
        m = x.shape[0]

        inv_sigma2_e = tf.exp(-2*self.log_sigma_e)
        inv_sigma2_w = tf.exp(-2*self.log_sigma_w)

        # We are not using the prediction y_hat of the model
        Phi_tilde, _ = self.joint_model(x)
        # Concat vector of ones to Phi_tilde
        Phi = tf.concat([Phi_tilde, tf.ones((m,1))], axis=1)
        
        Lambda_p_bar = self.get_Lambda_p_bar(Phi)

        # Get explicit solution of for w
        w_list = []
        y_list = []

        for i in range(self.n_y):
            Lambda_p_i = inv_sigma2_e[i]*Lambda_p_bar
            w_list.append(tf.linalg.solve(Lambda_p_i, inv_sigma2_e[i]*tf.linalg.matmul(tf.transpose(Phi),y[:,[i]])))
            y_list.append(tf.linalg.matmul(Phi,w_list[i]))

        w = tf.concat(w_list,axis=1)
        y_hat = tf.concat(y_list,axis=1)

        # Terms used in formulation of cost:
        dy = (y-y_hat)
        dy_square = tf.math.square(dy)
        w_square  = tf.math.square(w)

        # Add the individual terms to the negative log marginal likelihood
        J = 0
        J += self.n_y/(2)   * tf.math.log(2*np.pi)
        J += self.n_y/(2*m) * tf.linalg.logdet(Lambda_p_bar)
        J += self.n_y/(2*m) * (self.n_phi)*self.log_alpha

        inv_sigma2_w = tf.exp(-2*self.log_sigma_w)

        for i in range(self.n_y):  
            J += self.log_sigma_e[i]
            J += 1/(2*m) * tf.reduce_sum(dy_square[:,i])*inv_sigma2_e[i]
            J += 1/(2*m) * tf.reduce_sum(w_square[:,i]) *inv_sigma2_w[i]

        return J

    @tf.function
    def _train_step(self, x, y):
        """Perform a single training step.
        
        Computes the loss (based on :py:meth:`lml`) and gradients w.r.t to all ``trainable_variables``, and apply them to the model with the optimizer.
        The loss value is returned and can be stored to monitor training progression.

        This method has the ``@tf.function`` decorator for faster execution. For debugging it is advised to remove the decorator.

        Warning:
            This function is not intended to be called directly. Use :py:meth:`train` instead.
        """
        # Compute the loss and track the forward pass
        with tf.GradientTape() as tape:
            loss_value = self.lml(x, y, training=True)

        # From the tape, compute the gradients
        grads = tape.gradient(loss_value, self.trainable_variables)

        # Use the gradients to update the weights of the model with the optimizer
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return the loss value 
        return loss_value

    def _check_data_validity(self, x, y=None):
        """Check if the data is valid. Check either just inputs or inputs and outputs.

        Warning:
            This function is not intended to be called directly.

        Checks for:
            - Correct data type (numpy.ndarray or tf.Tensor)
            - Correct data shape (2D)
            - x and y have the correct dimension for the model
            - x and y have the same number of data points
        """
        if not isinstance(x, (np.ndarray, tf.Tensor)):
            raise TypeError('x must be a numpy.ndarray or a tf.Tensor')
        if len(x.shape) != 2:
            raise ValueError('x must be a 2D array.')
        if x.shape[1] != self.n_x:
            raise ValueError(f'x must have shape (m, n_x) with n_x = {self.n_x}.')
        if x.dtype != 'float32':
            raise TypeError('x must be a float32 array.')
        if y is not None:
            if not isinstance(y, (np.ndarray, tf.Tensor)):
                raise TypeError('y must be a numpy.ndarray or a tf.Tensor')
            if len(y.shape) != 2:
                raise ValueError('y must be a 2D array.')
            m_x = x.shape[0]
            m_y = y.shape[0]
            if m_x != m_y:
                raise ValueError(f'x ({m_x} samples) and y ({m_y} samples) must have the same number of data points.')
            if y.shape[1] != self.n_y:
                raise ValueError(f'y must have shape (m, n_y) with n_y = {self.n_y}.')
            if y.dtype != 'float32':
                raise TypeError('y must be a float32 array.')
    
    
    def fit(self, x, y, val=None, epochs=100, batch_size=None, verbose=False, callbacks=[], **kwargs):
        """Train the model on the given data.

        Internally, the method calls the :py:meth:`_train_step` method for each training step which in turn evaluates the loss function (based on :py:meth:`lml`) 
        and computes the gradients w.r.t to all ``trainable_variables``. The gradients are then applied to the model with the optimizer.

        Warning:
            batch training is not fully supported yet and currently raises a NotImplementedError.

        Note:
            The method automatically scaled the data. Don't pass scaled data.

        Args:
            x (tf.Tensor or numpy.ndarray): Input data of shape (m, n_x).
            y (tf.Tensor or numpy.ndarray): Output data of shape (m, n_y).
            val (tuple, optional): Tuple of validation data (x_val, y_val). Defaults to None.
            epochs (int, optional): Number of epochs to train the model. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to None, resulting in using the full dataset.
            verbose (bool, optional): Print training progress. Defaults to True.
            callbacks (list, optional): List of (keras) callbacks to be called during training. Defaults to [].

        Raises:
            NotImplementedError: If batch training is used.
            RuntimeError: If the model is not compiled.
            Error: If the data is not valid (see :py:meth:`_check_data_validity`).
            TypeError
        """
        # Sanity checks
        if self.flags['setup_training'] is False:
            raise RuntimeError('Call setup_training() first.')
        self._check_data_validity(x, y)
        if batch_size is not None:
            raise NotImplementedError('Batch training is not implemented yet.')
        if isinstance(val, (tuple, list)):
            self._check_data_validity(*val)
        if isinstance(callbacks, list):
            callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self.joint_model)
        else:
            raise TypeError('callbacks must be a list of callbacks or an empty list.')

        # Number of data points
        m = x.shape[0]

        # Scale the data
        x_scaled, y_scaled = self.scaler.scale(x, y)
        if val is not None:
            val_scaled = self.scaler.scale(*val)

        # batch_size defaults to m
        if batch_size is None:
            batch_size = m
        n_batches = m//batch_size

        # Shuffle and create batches
        ind = np.random.permutation(m)
        batch_ind = np.split(ind, [batch_size*k for k in range(1, n_batches)])

        x_batch_train = [x_scaled[batch_ind_k] for batch_ind_k in batch_ind]
        y_batch_train = [y_scaled[batch_ind_k] for batch_ind_k in batch_ind]

        # Train the model use keras callbacks.
        logs = {}
        callback_list.on_train_begin(logs=logs)
        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch, logs=logs)
            for batch, x_batch_k, y_batch_k in zip(range(n_batches), x_batch_train, y_batch_train):
                callback_list.on_batch_begin(batch, logs=logs)
                callback_list.on_train_batch_begin(batch, logs=logs)

                logs['loss'] = self._train_step(x_batch_k, y_batch_k)
                self.training_history['loss'].append(logs['loss'])
                self.training_history['epochs'].append(epoch)
                
                callback_list.on_train_batch_end(batch, logs=logs)
                callback_list.on_batch_end(batch, logs=logs)

            if val is not None:
                callback_list.on_batch_begin(batch, logs=logs)
                callback_list.on_test_batch_begin(batch, logs=logs)
                logs['val_loss'] = self.lml(val_scaled[0], val_scaled[1])
                self.training_history['val_loss'].append(logs['val_loss'])
                callback_list.on_test_batch_end(batch, logs=logs)
                callback_list.on_batch_end(batch, logs=logs)

            if verbose and val is not None:
                tools.print_percent_done(epoch, epochs, logs['loss'], logs['val_loss'])
            elif verbose:
                tools.print_percent_done(epoch, epochs, logs['loss'])

            callback_list.on_epoch_end(epoch, logs=logs)

            if self.joint_model.stop_training:
                break

        self.train_lml = self.lml(x_scaled, y_scaled).numpy() 

        callback_list.on_train_end(logs=logs)


class BayesianLastLayer(LogMarginalLikelihood):
    """Bayesian last layer model.

    The class :py:class:`BayesianLastLayer` extends a Keras model with a Bayesian last layer.
    The model is trained using the log marginal likelihood (:py:meth:`lml`) as loss function and
    yields a Gaussian posterior distribution over the weights of the last layer as well as the
    predictive distribution over the output data.

    The class must be initialized with a Keras model (``joint_model``) and a :py:class:`Scaler` object (``scaler``).
    The joint model must have two outputs, the first output is the output of the last (hidden) layer and the second output
    is the **linear transformation** of the last layer (the regular output of the model that should match the training data).

    The class inherits methods from :py:class:`LogMarginalLikelihood` that enable training the model with 
    the log marginal likelihood as loss function.

    The workflow for this class is as follows:

    1. Initialize the class with a Keras model and a :py:class:`Scaler` object.
    2. Prepare the training process with :py:meth:`prepare_training`.
    3. Train the model with the :py:meth:`train` method.
    4. Find an optimal value for alpha with :py:meth:`grid_search_alpha`.
    5. Update :py:attr:`log_alpha` with the optimal value.
    6. Make predictions with :py:meth:`predict`.

    The class has a ``__repr__`` method implemented and can be printed to the console for a summary of the model.

    Args:
        joint_model (keras.Model): Keras model with two outputs (last layer and linear transformation).
        scaler (Scaler): Scaler object for scaling the input and output data.

    Returns:
        BayesianLastLayer: Bayesian last layer model.

    """
    def __init__(self, joint_model, scaler, *args, **kwargs):
        # Sanity checks
        if len(joint_model.outputs) != 2:
            raise ValueError('The joint model must have two outputs. The first output is the feature space, the second output is the prediction.')
        if len(joint_model.inputs) != 1:
            raise ValueError('The joint model must have a single input.')
        if joint_model.layers[-1].get_config()['activation'] != 'linear':
            raise ValueError('The last layer must have a linear activation function.')

         # Store the joint model
        self.joint_model = joint_model
        # Number of inputs
        self.n_x = joint_model.inputs[0].shape[1]
        # Number of outputs
        self.n_y = joint_model.outputs[1].shape[1]
        # Number of features (considering the bias term):
        self.n_phi = joint_model.outputs[0].shape[1] + 1

        self.scaler = scaler
        if self.n_x != scaler.n_x:
            raise ValueError('The number of inputs of the joint model and the scaler must be the same.')
        if self.n_y != scaler.n_y:
            raise ValueError('The number of outputs of the joint model and the scaler must be the same.')

        # Initialize the LogMarginalLikelihood base class
        super().__init__(*args, **kwargs)

        # Update the flags dictionary (inherited from base class)
        self.flags.update({
            'prepare_prediction': False,
        })


    def __repr__(self):
        return_str = 'BayesianLastLayer\n'
        return_str += '----------------\n'
        return_str += 'State:\n'
        for key in self.flags.keys():
            return_str += '- {} = {}\n'.format(key, self.flags[key])
        return_str += '- n_x   = {}\n'.format(self.n_x)
        return_str += '- n_phi = {}\n'.format(self.n_phi)
        return_str += '- n_y   = {}\n'.format(self.n_y)
        if self.flags['prepare_prediction']:
            return_str += 'Results:\n'
            return_str += f'- train_lml   = {np.round(np.atleast_1d(self.train_lml), 3)}\n'
            return_str += f'- log_alpha   = {np.round(np.atleast_1d(self.log_alpha.numpy()), 3)}\n'
            return_str += f'- log_sigma_e = {np.round(np.atleast_1d(self.log_sigma_e.numpy()), 3)}\n'

        return return_str


    def fit(self, x, y, *args, **kwargs):
        """
        Calls the :py:meth:`fit` method of the base class :py:class:`LogMarginalLikelihood`.

        After successful training, the method :py:meth:`prepare_prediction` is called to compute the posterior precision matrix and the covariance matrix of the weights.
        This allows to call the :py:meth:`predict` method. 
        """
        super().fit(x, y, *args, **kwargs)
        self.prepare_prediction(x)
        
        
    def prepare_prediction(self, X_train):
        """Prepare the prediction by computing the posterior precision matrix and the covariance matrix of the weights.
        These matrices do not change after training and should only be computed once to accelerate the call of of :py:meth:`predict`.

        This method is called from :py:meth:`train` after training the model with the training data.
        The user only needs to call this method to update the training data which is typically not advised.

        Args:
            X_train (np.ndarray): The original training data that was used to train the model. The data should be unscaled.

        Retruns:
            None
        """

        # Number of samples
        self.m = X_train.shape[0]

        # Scale the data
        self.X_scaled = self.scaler.scale(X = X_train)[0]

        Phi = self.joint_model(self.X_scaled)[0].numpy()
        Phi = np.concatenate([Phi, np.ones((self.m, 1))], axis=1)
        self.Phi = Phi.astype('float32')

        self.Lambda_p_bar = self.get_Lambda_p_bar(self.Phi).numpy()
        self.Sigma_p_bar = np.linalg.pinv(self.Lambda_p_bar)

        # Extract Sigma_e
        sigma_e2 = np.exp(2*self.log_sigma_e.numpy())
        self.Sigma_e = np.diag(sigma_e2)
        self.Sigma_e_inv = np.diag(1/sigma_e2)

        self.flags['prepare_prediction'] = True

    def score(self, X, Y, scoring, verbose=False, *args, **kwargs):
        """
        Compute the score of the model on the given data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Output data.
            scoring (str): Scoring method. Can be either ``lml``, ``mse``, ``lpd`` or ``pd``.
                - ``lml``: Log marginal likelihood.
                - ``mse``: Mean squared error.
                - ``lpd``: Log predictive density.
                - ``pd``: Predictive density.
            verbose (bool): If True, print the score.

        Returns:
            float: Score of the model.
        """

        if scoring == 'lml':
            X_scaled, Y_scaled = self.scaler.scale(X, Y) 
            score = self.lml(X_scaled, Y_scaled).numpy()
        elif scoring == 'mse':
            score = self.mse(X, Y)
        elif scoring == 'lpd':
            score = self.lpd(X, Y, *args, **kwargs)
        elif scoring == 'pd':
            score = self.predictivedensity(X, Y, *args, **kwargs)
        else:
            raise ValueError(f'Scoring method {scoring} is not supported')

        if verbose:
            print(f'score ({scoring}): {np.round(score,3)}')

        return score


    def predict(self, x, uncert_type='cov', return_scaled = False, with_noise_variance = False, return_grad = False):
        """
        Predict the output of the model for a given input x.
        This evaluates the neural network model and computes the covariance or standard devaiation of the predictive distribution.

        Predict can also return the gradient of the mean function with respect to the input data. This is useful for sensitivity analysis and uncertainty propagation.
        The gradient can only be computed if m = 1 (a single sample is given as input).

        Args:
            x: Input data (numpy array of shape (m, n_x))
            uncert_type: Type of uncertainty to return. Can be 'cov' for the covariance matrix or 'std' for the standard deviation or 'none' for no uncertainty.
            return_scaled: Return the scaled (normalized) values of th prediction for mean and uncertainty.
            with_noise_variance: Add the (estimated) noise variance to the predictive covariance matrix. This is useful when comparing predictions to (noisy) test data, e.g. for the :py:meth:`score` method with ``scoring=lpd``. 
            return_grad: Return the gradient of the mean function with respect to the input data.

        Returns:
            y_pred: Predicted output (numpy array of shape (m, n_y))
            y_uncert (optional): Uncertainty of the prediction. In case of 'cov' this is the covariance matrix (numpy array of shape (m*n_y, m*n_y)). For 'std' this is the standard deviation (numpy array of shape (m, n_y))
            grad (optional): Gradient of the mean function with respect to the input data (numpy array of shape (n_y, n_x))
        """
        # Sanity checks
        if not self.flags['prepare_prediction']:
            raise RuntimeError('You need to call the prepare_prediction method first.')
        self._check_data_validity(x)

        if return_grad: 
            y_hat, phi, grad = self.mean(x, return_scaled, return_grad)
        else:
            y_hat, phi = self.mean(x, return_scaled, return_grad)

        y_hat_covariance = self.covariance(phi, self.Sigma_p_bar, return_scaled, with_noise_variance)

        # Store phi for debugging
        self.phi = phi

        out = [y_hat]

        if uncert_type == 'cov':
            out.append(y_hat_covariance)
        elif uncert_type == 'std':
            y_hat_std = self.std(y_hat_covariance)
            out.append(y_hat_std)
        elif uncert_type == 'none':
            pass
        else:
            raise ValueError(f'Uncertainty type {uncert_type} is not supported.')

        if return_grad:
            out.append(grad)

        return out

    def mean(self, x, return_scaled = False, return_grad = False):
        """
        Compute the mean of the predictive distribution.
        This method is also called from the :py:meth:`predict` method.

        Args:
            x: Input data (numpy array of shape (m, n_x))

        Returns:
            y_hat_mean: Mean of the predictive distribution (numpy array of shape (m, n_y))
            phi: Feature vector (numpy array of shape (m, n_phi))
        """
        m_test = x.shape[0]
        x_scaled = self.scaler.scale(X = x)[0]

        if return_grad:
            assert x.shape[0] == 1, 'Gradient computation only supported for a single test point.'
            # For tracing we need to wrap x in a tf.Variable (only neccessary for gradient computation)
            x_scaled = tf.Variable(x_scaled)
        
        with tf.GradientTape() as tape:
            phi_tilde, y_hat_scaled = self.joint_model(x_scaled)
        
        if return_grad:
            grad = tape.jacobian(y_hat_scaled, x_scaled)
            grad = tf.squeeze(grad).numpy()
            grad = np.diag(self.scaler.scaler_y.scale_)@(grad)

        # Convert to numpy 
        phi_tilde = phi_tilde.numpy()
        y_hat_scaled = y_hat_scaled.numpy()

        phi = np.concatenate((phi_tilde, np.ones((m_test,1))), axis=1)

        if return_scaled:
            out = [y_hat_scaled, phi]
        else:
            y_hat_unscaled = self.scaler.unscale(Y = y_hat_scaled)[0]
            out = [y_hat_unscaled, phi]

        if return_grad:
            out.append(grad)

        return out

    def covariance(self, phi, Sigma_p_bar, return_scaled = False, with_noise_variance = False):
        """Compute the covariance matrix of the predictive distribution.

        Warning:
            This method is not intended to be called directly. Use :py:meth:`predict` instead.

        Args:
            phi (np.ndarray): The feature space (numpy array of shape (m, n_phi))
        
        Returns:
            cov (np.ndarray): The covariance matrix of the predictive distribution (numpy array of shape (m*n_y, m*n_y))
        """

        m_test = phi.shape[0]

        # Get the variances of the noise for the individual outputs (n_y, 1) vector
        sigma_e2 = np.exp(2*self.log_sigma_e.numpy())

        # Covariances are computed independently for each output
        cov_list = []
        # Baseline covariance for all outputs
        cov_0 = (phi @ Sigma_p_bar @ phi.T)

        for i, sigma_e2_i in enumerate(sigma_e2):
            # Get covariance of the i-th output. The covariance is still scaled.
            cov_i_scaled = sigma_e2_i * cov_0 
            if with_noise_variance:
                cov_i_scaled += sigma_e2_i * np.eye(m_test)

            # Unscale the covariance of the i-th output. Consider rules of linear transformation of Gaussian variables 
            # e.g. https://services.math.duke.edu/~wka/math135/gaussian.pdf
            if return_scaled:
                cov_list.append(cov_i_scaled)
            else:
                cov_i = cov_i_scaled * self.scaler.scaler_y.scale_[i]**2
                cov_list.append(cov_i)

        # To obtain the full (m*n_y, m*n_y) covariance matrix, stack the individual covariances as a blockdiagonal matrix.
        y_hat_covariance = tools.blkdiag(*cov_list)

        return y_hat_covariance

    def std(self, Cov):
        """Convert the covariance matrix to standard deviation.

        The standard deviation is reshaped to ``(m, n_y)`` to match the shape of the predictions.
        
        Warning:
            This method is not intended to be called directly. Use :py:meth:`predict` instead.

        Args:
            Cov (np.ndarray): The covariance matrix of the predictive distribution (numpy array of shape ``(m*n_y, m*n_y)``)

        Returns:
            std (np.ndarray): The standard deviation of the predictive distribution (numpy array of shape ``(m, n_y)``)
        """
        y_hat_std = np.sqrt(np.diag(Cov))
        y_hat_std = np.reshape(y_hat_std, (-1, self.n_y), order='F')

        return y_hat_std

    def predictivedensity(self, X, y,  aggregate = 'mean'):
        """Computes the predictive density instead of the log predictive density.
        See documentation of :meth:`lpd` for more information.
        """

        lpd = self.lpd(X, y, aggregate="none")
        pd = np.exp(lpd)

        if aggregate == 'none':
            return pd
        if aggregate == 'median':
            return np.median(pd).flatten()
        elif aggregate == 'mean':
            return np.mean(pd).flatten()
        else:
            raise ValueError(f'Unknown aggregation method {aggregate}') 

    def lpd(self, X, y, aggregate='mean'):
        """
        Computes the log probability density of observing ``y_hat`` given ``x`` and ``y``
        """
        # Sanity checks
        self._check_data_validity(X, y)
        # Scale y
        _, y_scaled = self.scaler.scale(X,y)

        m_t = X.shape[0]

        y_pred_scaled, Phi = self.mean(X, return_scaled=True)

        # Difference between true and predicted targets (m, n_y)
        dY = y_scaled - y_pred_scaled # type:ignore

        Sigma_y_bar = Phi@self.Sigma_p_bar@Phi.T
        sigma_y_bar = np.diag(Sigma_y_bar)+1 # add 1 to account for noise variance 
        Sigma_y_bar = np.diag(sigma_y_bar)
        Lambda_y_bar = np.diag(1/sigma_y_bar)

        logp = -.5*self.n_y*np.log(sigma_y_bar)
        logp+= -.5*np.diag(Lambda_y_bar@dY@self.Sigma_e_inv@dY.T)
        logp+= -.5*np.linalg.slogdet(2*np.pi*self.Sigma_e)[1]

        if aggregate == 'none':
            return logp
        if aggregate == 'median':
            return np.median(logp).flatten()
        elif aggregate == 'mean':
            return np.mean(logp).flatten()
        else:
            raise ValueError(f'Unknown aggregation method {aggregate}') 


    def mse(self, X, y):
        """
        Computes the mean squared error of the model
        """
        # Sanity checks
        self._check_data_validity(X, y)
        y_scaled = self.scaler.scale(Y = y)[0]
        y_pred_scaled, _ = self.predict(X, uncert_type='std', return_scaled=True)
        mse = np.mean(np.square(y_pred_scaled - y_scaled)).flatten()
        return mse


    def grid_search_alpha(self, x, y, rel_range=[0,10], scores = ['lpd'], samples=10, max_cond = 1e8, verbose=True, average=True):
        """Simple grid search to test different values of alpha.
        The search evaluates the predictive log-probability of the posterior distribution
        for a test set of data (x, y) which must not have been used for training.

        The method stops testing alpha if the condtion number of the posterior precision matrix exceeds
        ``max_cond``. 

        Args:
            x (tf.Tensor or numpy.ndarray): Input data of shape (m, n_x).
            y (tf.Tensor or numpy.ndarray): Output data of shape (m, n_y).
            rel_range (list, optional): Range of alpha values to test relative to optimal alpha*. Defaults to [0,10].
            samples (int, optional): Number of samples to evaluate the predictive log-probability. Defaults to 10.
            max_cond (float, optional): Maximum condition number of the posterior precision matrix. Defaults to 1e8.
            verbose (bool, optional): Print training progress. Defaults to True.

        Returns:
            log_alpha_test (numpy.ndarray): Array of tested log_alpha values of shape (samples,).
            logprob_test (numpy.ndarray): Array of predictive log-probabilities for the test data of shape (samples,).
        """

        log_alpha_min = float(self.log_alpha)+rel_range[0]
        log_alpha_max = log_alpha_min +rel_range[1]
        log_alpha_opt = copy.copy(self.log_alpha.numpy())

        log_alpha_test = np.linspace(log_alpha_min, log_alpha_max, samples).astype('float32')
        y_hat_unscaled, phi = self.mean(x) 

        x_scaled, y_scaled = self.scaler.scale(x,y)

        results = {
            key: [] for key in scores
            }

        for k, log_alpha in enumerate(log_alpha_test):
            self.log_alpha = log_alpha

            Lambda_p_bar = self.get_Lambda_p_bar(self.Phi)

            if np.linalg.cond(Lambda_p_bar)>max_cond:
                log_alpha_test = log_alpha_test[:k]
                break
            else:
                for score in scores:
                    results[score].append(self.score(x, y, scoring=score))

            if verbose:
                tools.print_percent_done(k, samples, title='Testing alpha')

        for score in scores:
            results[score] = np.array(results[score]).flatten() # type:ignore

        results['log_alpha'] = log_alpha_test                   # type:ignore
        
        # Reset to optimal alpha
        self.log_alpha = log_alpha_opt

        return results

    def save(self, path, custom_objects={}):
        """
        Saves the model to a file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.custom_objects = custom_objects

        with open(path, 'wb') as f:
            pickle.dump(self, f)


    def __getstate__(self):
        """Return state values to be pickled.
        The method is only necessary for pickling the class.
        """
        state = self.__dict__.copy()
        model_weights = self.joint_model.get_weights()
        model_config = self.joint_model.get_config()
        state['model_weights'] = model_weights
        state['model_config'] = model_config
        # Remove model, optimizer and trainable variables
        state.pop('joint_model')
        state.pop('optimizer','')
        state.pop('trainable_variables','')

        #Update flags
        state['flags']['setup_training'] = False

        return state

    def __setstate__(self, state):
        """Set state values from pickle.
        The method is only necessary for unpickling the class.
        """
        # Retrieve model from weights and config
        with keras.utils.custom_object_scope(state['custom_objects']):
            joint_model = keras.models.Model.from_config(state['model_config'])

        joint_model.set_weights(state['model_weights'])
        state['joint_model'] = joint_model
        # Remove weights and config from state
        state.pop('model_weights')
        state.pop('model_config')
        # Set state
        self.__dict__.update(state)


# Functions used for testing the class.


def get_model(n_in,n_out, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_input = keras.Input(shape=(n_in,))

    # Hidden units
    architecture = [
    (keras.layers.Dense, {'units': 20, 'activation': tf.math.sin, 'name': '01_dense'}),
    (keras.layers.Dense, {'units': 20, 'activation': tf.nn.tanh, 'name': '02_dense'}),
    (keras.layers.Dense, {'name': 'output', 'units': n_out})
    ]

    # Get layers and outputs:
    model_layers, model_outputs = tools.DNN_from_architecture(model_input, architecture)
    output_model = keras.Model(inputs=model_input, outputs=model_outputs[-1])
    joint_model = keras.Model(model_input, [model_outputs[-2], model_outputs[-1]])
    return joint_model, output_model


def test():
    """
    Test the main functionalities of the class :class:`BayesianRegression`.

    - Create data and train the class with :meth:`BayesianRegression.fit`.
    - Check for better values of alpha with :meth:`BayesianRegression.grid_search_alpha`.
    """


    import tools
    import matplotlib.pyplot as plt
        
    n_samples = 100
    seed = 99

    function_types = [2, 0]
    sigma_noise = [2e-2, 5e-1]
    n_channels = len(function_types)

    data = tools.get_data(n_samples,[0,1], function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
    test = tools.get_data(20, [-.2,1.2],   function_type=function_types, sigma=sigma_noise, dtype='float32', random_seed=seed)
    true = tools.get_data(300, [-.2,1.2],  function_type=function_types, sigma=[0.,0.], dtype='float32')

    train, val = tools.split(data, test_size=0.2)

    # Create scaler from training data
    scaler  = tools.Scaler(*train)

    # Scale data (only required for testing purposes)
    train_scaled = scaler.scale(*train)
    val_scaled = scaler.scale(*val)
    test_scaled = scaler.scale(*test)
    true_scaled = scaler.scale(*true)
    # Fix seeds
    seed = 99
    # Create model
    n_in = data[0].shape[1]
    n_out = data[1].shape[1]
    joint_model, output_model = get_model(n_in, n_out, seed)

    # Get bll model
    bllmodel = BayesianLastLayer(joint_model, scaler)

    # Pickle model
    bllmodel.save('results/test.pkl', custom_objects={'sin': tf.math.sin})
    
    # Load model
    with open('results/test.pkl', 'rb') as f:
        bllmodel = pickle.load(f)

    #Prepare training
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    #Setup training:
    bllmodel.setup_training(optimizer)

    # Train model
    if True:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Get callback for early stopping
        cb_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=100,
        verbose=True,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        )

        # Train
        bllmodel.fit(*train, val=val, epochs=100, verbose=True, callbacks=[cb_early_stopping])


    print(bllmodel)

    # Get some scores:
    score_str = f'Training score:\n'
    score_str += f'Log marginal likelihood: {bllmodel.score(*train, "lml")}'
    score_str += f'Log marginal likelihood: {bllmodel.score(*train, "lpd")}'
    score_str += f'Log marginal likelihood: {bllmodel.score(*train, "mse")}'
    print(score_str)


    res_grid_search_alpha = bllmodel.grid_search_alpha(*test, rel_range=[-3,15], scores=['lpd', 'lml'], samples = 50, max_cond=1e8)

    fig, ax = plt.subplots(2)
    ax[0].plot(res_grid_search_alpha['log_alpha'], res_grid_search_alpha['lpd'])
    ax[0].set_xlabel('log(alpha)')
    ax[0].set_ylabel('logprob')
    ax[1].plot(res_grid_search_alpha['log_alpha'], res_grid_search_alpha['lml'])
    ax[1].set_xlabel('log(alpha)')
    ax[1].set_ylabel('lml')
    fig.tight_layout()
    fig.show()

if __name__ == "__main__":
    test()