# ridgeRegression-with-Keras
Regularised Ridge Regression with Keras


I was trying to perform a ridge regression with python.
Matrices were:
X: (32496, 42309)
y: (32496, 1)
 
Since, I had to perform ~4000 of them, and optimise everything (i.e. optimise regularisation parameter for every regression) I had to do it pretty quickly.
With [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), really useful python library for machine learning, takes a day for every regression.
I tried also this other library, [h2o4gpu](https://github.com/h2oai/h2o4gpu), that it’s essentially a porting to GPU of sklearn: very nice, well done, incredibly fast, but for ridge regression there is a [bug](https://github.com/h2oai/h2o4gpu/issues/175), and you cannot run a loop of regressions.

This is why I decided to use [keras](https://keras.io/), a high-level deep learning framework.
You can easily implement a simple regression and add reguralisation.
I did an example code to explain that. In this code you don’t see a big difference in time, but in a real application there is!
Result: I can obtain a good approximation within ~10/20 sec., instead of 1 day. 👍🏻
The cons are: you need a gpu, you need to install deep learning libraries, and the training can be quite tricky.
