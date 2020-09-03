# lasso regression

In this tutorial, we will use a simple linear model **AX=B** for classification and use sparsity-promoting technique for salient feature identification. The tutorial is intended for beginners.

## How easy is it to distinguish MNIST images?

In [this Jupyter Notebook](https://github.com/rsyamil/lasso-regression/blob/master/linear_model_for_classification.ipynb), we will use the digit-MNIST dataset from Keras. Here are some sample images for each class label and to a human, they don't appear to be too challenging to distinguish given that these are decent hand-written digits. 

![Dataset](/readme/dataset.png)

From [this Kaggle discussion forum](https://www.kaggle.com/c/digit-recognizer/discussion/61480), we see that recent advanced algorithms are able to obtain over 97% accuracy on the digit MNIST dataset. If we look at the left side of the histogram, linear models are able to achieve up to 92% accuracy.

![Leaderboard](/readme/leaderboard.png)

In this tutorial, we will use a simple linear model **AX=B** to classify MNIST digits. Specifically, each of the images is flattened into a long vector and each of the corresponding labels is represented as a one-hot encoding vector. **X** is the set of flattened images and **B** is the set of one-hot encoding vectors while **A** represents the weights to be learnt in the linear model. There are many linear solvers that can be used but for the purpose of this demonstration, we will use Keras. The system of linear equations are solved as a regression problem. 

![Architecture](/readme/architecture.png)

Once the weights are learnt, we obtain the prediction by simply multiplying the learnt weights with the training or testing images. Here are some sample predictions and note that for each image, the prediction vector cannot be viewed as the likelihood. We can however take the *argmax()* of the prediction vector to identify the most relevant class label. Note that the prediction vector for an image can also contain negative values!

![Predictions](/readme/predictions.png)

Using this simple method, we are able to obtain ~85% accuracy on both the training and testing datasets. It seems that it is not that hard afterall to distinguish the digits! 

## Sparsity-promoting linear model for salient feature identification

Next, in [this Jupyter Notebook](https://github.com/rsyamil/lasso-regression/blob/master/linear_model_for_classification_sparse.ipynb), we will apply L1 regularization on the weight matrix. In Keras, this can easily be done by specifying the kernel regularizer. L1 regularization is useful in this case as it will kill small irrelevant weights in **A** and those weights will remain zero. An important parameter to tune is the regularization penalty, *lambda*. A sensitivity analysis shows that *lambda* of *1e-3* will give us about 81% accuracy while still giving a sparse solution of **A**. 

![Weights_sparse_sensitivity](/readme/weights_sparse_sensitivity.png)

Suppose we pick *lambda* of *1e-3* as the suitable L1 penalty, we then apply a threshold on the learnt weights to obtain a sparse mask. Any value in **A** that is smaller than the threshold will correspond to a value of *0* in the sparse mask, and *1* otherwise. In the notebook, we define a *Dense* layer that allows element-wise multiplication with a mask. When this sparse mask is provided to **A**, any small values below the threshold will be multiplied with a *0* and will not contribute to the prediction vectors in **B**.

![Weight_thresholding](/readme/weight_thresholding.png)

The sparse mask represents the important pixels that the linear model will look at in calculating the prediction vector (which is essentially the sum of the linear combination of the said pixels). Each column (with dimension of *784*) in **A** represents the important pixels for each class label. Lets take each column, reshape it into a 2D image and plot it on top of the mean (for each class label) of the training images.

![Salient_features](/readme/salient_features.png)

We can see that the important pixels agree with the salient identifying features for each of the class labels. For example, a *seven* is easily identifiable if there exist two lines meeting at an acute angle and there is nothing written between these two lines (plot the magnitude of the weights to see this).

So, how easy is it to distinguish MNIST digits? Quite easy I'd say. Then again the dataset is rather simple and this is just a tutorial to demonstrate what you can do with lasso regression. 
