import os 
os.environ["KERAS_BACKEND"]="jax"

import jax 
import jax.numpy as jnp 
import keras
import matplotlib.pyplot as plt 
import numpy as np

class BNN_Layer(keras.layers.Layer):
    def __init__(self, units=1,prior_mu=0.0,prior_std=1.0,seed=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.prior_mu = prior_mu
        self.prior_std = prior_std
        self.seed=keras.random.SeedGenerator(seed)

    def build(self, input_shape):
        # w
        # self.w = None  # shape: (input_shape[-1],units)
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.w_rho = self.add_weight(shape=(input_shape[-1], self.units), initializer="zeros", trainable=True)

        # b
        # self.b = None  # shape: (units,)
        self.b_mu = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.b_rho = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

        self.built = True

    def _KL(self, sigma_q, mu_q, sigma_p, mu_p):
        var_q = sigma_q**2
        var_p = sigma_p**2
        # return 0.5 * (jnp.log(var_p) - jnp.log(var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        term1 = keras.ops.log(sigma_p) - keras.ops.log(sigma_q)
        term2 = (var_q + (mu_q - mu_p) ** 2) / (2 * var_p)
        return term1 + term2 - 0.5

    def call(self, x):
        # w_mu = jnp.array(self.w_mu)
        # w_rho = jnp.array(self.w_rho)
        # b_mu = jnp.array(self.b_mu)
        # b_rho = jnp.array(self.b_rho)

        # w
        # w_epsilon = jax.random.normal(jax.random.key(11), shape=w_mu.shape)
        # w_epsilon = np.random.normal(size=w_mu.shape)
        w_epsilon = keras.random.normal(shape=self.w_mu.shape,seed=self.seed)
        w_sigma = keras.ops.log(1 + keras.ops.exp(self.w_rho))
        w = self.w_mu + w_sigma * w_epsilon

        # b
        # b_epsilon = jax.random.normal(jax.random.key(seed=np.random.randint(0,100)), shape=b_mu.shape)
        # b_epsilon = np.random.normal(size=b_mu.shape)
        b_epsilon = keras.random.normal(shape=self.b_mu.shape,seed=self.seed)
        b_sigma = keras.ops.log(1 + keras.ops.exp(self.b_rho))
        b = self.b_mu + b_sigma * b_epsilon

        y = keras.ops.matmul(x, w) + b  # predict output
        kl_w = keras.ops.sum(self._KL(w_sigma, self.w_mu, self.prior_std, self.prior_mu))  # KL of w
        kl_b = keras.ops.sum(self._KL(b_sigma, self.b_mu, self.prior_std, self.prior_mu))  # kl of b

        kl = (kl_w + kl_b) / (w.size + b.size)*0.01

        self.add_loss(kl)
        return y


class BNN_Module:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        x = keras.Input(shape=self.input_shape)
        y = BNN_Layer(units=8,prior_std=10.0)(x)
        y = keras.activations.relu(y)
        y = BNN_Layer(units=8,prior_std=10.0)(x)
        y = keras.activations.relu(y)
        out = BNN_Layer(units=1,prior_std=10.0)(y)
        model = keras.Model(inputs=x, outputs=out)
        return model

    def predict(self, x, sample_num=100, verbose=0):
        y = []
        for _ in range(sample_num):
            tmp = self.model.predict(x, verbose=verbose)
            y.append(tmp)
        y = keras.ops.array(y)
        y_mean = keras.ops.mean(y, axis=0)
        y_var = keras.ops.var(y, axis=0)
        y_max = keras.ops.max(y, axis=0)
        y_min = keras.ops.min(y, axis=0)
        return y_mean, y_var, y_max, y_min

def test1(x):
    return -(x**4) + 3 * x**2 + 1


def test2(x):
    (x - 3.5) * keras.ops.sin((x - 3.5) / (np.pi))


def test3(x):
    y = x**3 - x**2 + 3 * np.random.rand(*x.shape)
    return y


test_fun = test1
x_train = keras.ops.array([-2, -1.8, -1, 1, 1.8, 2]).reshape(-1, 1)
y_train = test_fun(x_train)
print(y_train.shape)

model = BNN_Module(input_shape=(1,))
model.model.summary()
model.model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse", metrics=[keras.metrics.R2Score])
hist = model.model.fit(
    x=x_train,
    y=y_train,
    epochs=4000,
    callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=1000, restore_best_weights=True)],
)

x_test=keras.ops.linspace(-5.0,5.0,100).reshape(-1,1)
y_test=model.predict(x_test)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(x_train[:,0],y_train[:,0],"o",label="train points")
ax.plot(x_test[:,0],y_test[0][:,0],label="pred")
# ax.fill_between(x_test[:,0],y1=y_test[2][:,0],y2=y_test[3][:,0],alpha=0.25,label="max-min Confidence")
ax.fill_between(x_test[:,0],y1=y_test[0][:,0]-1.96*keras.ops.sqrt(y_test[1][:,0]),y2=y_test[0][:,0]+1.96*keras.ops.sqrt(y_test[1][:,0]),alpha=0.25,label="95% Confidence")
ax.legend()
plt.show()

