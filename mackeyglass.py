import tqdm
import FWESN
import numpy
import matplotlib.pyplot
import reservoirpy.datasets

D = 4
delta = 6
P = 85
n_timesteps=6000

# generate the Mackey-Glass distribution
mackey_glass = reservoirpy.datasets.mackey_glass(
    n_timesteps=n_timesteps,
    tau=17,
    a=0.2,
    b=0.1,
    x0=1.2
)   # (n_timesteps, 1)
print(f"Max: {numpy.max(mackey_glass)}")
print(f"Min: {numpy.min(mackey_glass)}")

Y = mackey_glass[delta*(D - 1)+P:]
X = mackey_glass[delta*(D - 1):][:Y.shape[0], :]
# break into train and test arrays
X_train, X_test = mackey_glass[:3000], mackey_glass[5000:]
Y_train, Y_test = mackey_glass[:3000], mackey_glass[5000:]

# create the fuzzy weighted echo state network
ofwesn = FWESN.eFWESN(
    dim_in=1,
    dim_res=8,
    dim_out=1,
    cauchy_r=0.1,
    firing_th=0.01,
    spectral_r=0.9
)

# fit the model to the training set
print("Training")
train_output = numpy.zeros(shape=X_train.shape)
for i in tqdm.tqdm(range(X_train.shape[0])):
    train_output[i] = ofwesn.run(
        X_train[i],
        Y_train[i]
    )[0]
print(f"Trained with {ofwesn.rules.shape[0]} rule(s)")

# reset state for testing
ofwesn.reserviour_state_reset()

# test the fitted model
test_output = numpy.zeros(shape=X_test.shape)
for i in tqdm.tqdm(range(X_test.shape[0])):
    test_output[i] = ofwesn.run(
        X_test[i],
        None
    )[0]


# plot the results
matplotlib.pyplot.plot(Y_train, label="Actual Y", color='red')
matplotlib.pyplot.plot(train_output, label="Predict Y", color='blue')
matplotlib.pyplot.ylabel("Training Output")
matplotlib.pyplot.xlabel("Training Input")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

matplotlib.pyplot.plot(Y_test, label="Actual Y", color='red')
matplotlib.pyplot.plot(test_output, label="Predict Y", color='blue')
matplotlib.pyplot.ylabel("Testing Output")
matplotlib.pyplot.xlabel("Testing Input")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
