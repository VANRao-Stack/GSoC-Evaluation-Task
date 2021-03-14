Storage is one of the main limiting factors to recording information from proton-proton collision events at the Large Hadron Collider at CERN in Geneva. Hence, the ATLAS experiment at the LHC uses a so-called trigger system, which selects and sends interesting events to the data storage system while throwing away the rest.
In this evaluation task, I have tried to alleviate the problem by writing an Autoencoder that compresses the 4-D PhenoML data to 3-D. 
This documentation covers everything I have done and explains everything down to the last detail. 

# Introduction 
## Autoencoders
The reader is assumed to be familiar with what a neural network is; however, here is a brief overview if this is not the case. 
A neural network can be assumed to be a stack of layers that performs some nonlinear operation on the input it receives. The question now becomes, what nonlinear operation is it?
A layer begins with applying some linear transformation on the input vector, as depicted below,
{matrix multiplication}
The transformed input vector is then subjected to a nonlinear activation function which induces the nonlinear characteristic of the layer as follows:
{activation function}
These layers then can be stacked on top of one another, and a network with more than two layers is called a deep neural network. 
Now, what is an Autoencoder, and how does it help compress data?
An autoencoder is a neural network that tries to reconstruct its input.
So if we feed the autoencoder the vector (1,0,0,1,0), the autoencoder will try to output (1,0,0,1,0). How the autoencoder performs this operation is the key to understanding how it helps us compress the data. 
Let us understand this using an instance. 
Say we have inputs in 5 dimensions as in our example, if we use two neurons in the hidden layer, our autoencoder will receive five features and "encode" them in 2 features in a way such as it can reconstruct the same five-dimensional input.
So we go from (1,0,0,1,0) to (x,y) and from (x,y) to (1,0,0,1,0).
This effectively means that (x,y) contains all information present in the input since we can reconstruct the input from (x,y). 
Hence instead of having to store an entire dataset of five-dimensional data-points, we could, in theory, save the compressed data-points along with the decoder to recover the original data-points. 

## The Dataset
The format of CSV files is: event ID; process ID; event weight; MET; METphi; obj1, E1, pt1, eta1, phi1; obj2, E2, pt2, eta2, phi2; . . .
Of which we need to concern ourselves with only obj1, E1, pt1, eta1, and phi1. 
Once we have the dataset imported, we begin by removing the unnecessary columns and retaining only the ones mentioned above. 
We follow this up by cleaning the dataset, i.e., making sure that the dataset has no erroneous values. 
Once this is done, we move to pre-process the data. 

# Pre-Processing 
In analyzing a particle's four-momentum, we can observe that E1 and pT1 were skewed, whereas the eta1 was almost gaussian shaped. It is hard to describe the exact shape of phi1, but it was close to being uniform. I proceeded to standardize it, after which it looked something like this.

![alt text](https://github.com/VANRao-Stack/gsoc_eval_task/blob/main/GSoC/WhatsApp%20Image%202021-03-13%20at%2023.18.08.jpeg)

Numerical input variables may have a highly skewed or non-standard distribution.
This could be caused by outliers in the data, multi-modal distributions, highly exponential distributions, and more.
In our case, we notice that the histogram of E1 and pT1 are both highly right-skewed. 
Many machine learning algorithms, including Neural Networks, prefer or perform better when numerical input variables and even output variables in the case of regression have a standard probability distribution, such as a Gaussian (normal) or a uniform distribution.
We see that E1 and pT1 are still skewed. One thing that was suggested as part of Eric Wulff's thesis was using logarithmic transform on the two columns to normalize them. Performing the suggested transformations and then standardizing results in the following dataset.

![alt text](https://github.com/VANRao-Stack/gsoc_eval_task/blob/main/GSoC/Screenshot%202021-03-13%20233256.png)

Although this does normalize the data, I also tried using the Quantile transformation to normalize the data. 
A quantile transform will map a variable's probability distribution to another probability distribution.
Recall that a quantile function, also called a percent-point function (PPF), is the inverse of the cumulative probability distribution (CDF). A CDF is a function that returns the probability of a value at or below a given value. The PPF is the inverse of this function and returns the value at or below a given probability.
The quantile function ranks or smooths out the relationship between observations and can be mapped onto other distributions, such as the uniform or normal distribution.
We map our data to the Gaussian distribution and standardize it; this results in the following dataset. 

![alt text](https://github.com/VANRao-Stack/gsoc_eval_task/blob/main/GSoC/Screenshot%202021-03-14%20013932.png)

To summarize, we created two versions of the same dataset, 
A dataset, as described in Eric Wulff's thesis, called the standard dataset henceforth. 
A dataset formed after applying the Gaussian Quantile transformation called the Rank Gauss dataset henceforth. 

# Autoencoder Architecture
We use the same architecture as described in Eric Wulff's thesis for testing. It contains eight layers, of which four belong to the encoder and four to the decoder. 
The layer size are [200,100,50,3,50,100,200]. 
The activation function (the nonlinear function used after the linear transformation) is the hyperbolic tangent function. 
We PyTorch to build the Network and use fastai to train the same. 

## Optimizer
As seen earlier, each layer in a neural network consists of a linear transformation followed by a nonlinear activation. The parameters of the linear transformation are called weights and are what is figured during the training process. 
We have multiple algorithms that help us find these parameters, but broadly we can characterize them based on the order of gradient we need to compute for the training process. 
We first have first-order optimizers, such as Adam, RMSProp, and SGD, that, as the name suggests, rely on first-order gradients to compute the optimal weights. 
We also have second-order optimizers such as the L-BFGS or the CG method that rely on the second-order gradients for computing the weights. 
I have utilized both the optimizers for training the Network described above. However, L-BFGS seemed to work exceptionally poorly, and hence I will ignore it for the rest of our discussion. 
The first order optimizer I have used is the same as that described in Eric Wulff's thesis. The Adam optimizer with weight decay. 
I noticed that the Network starts to overfit the data after processing around 3200 batches. Overfitting generally means that instead of learning a general representation of the data, the Network starts memorizing the training data; this leads to terrible performances on data it has not seen, which, as you would have guessed, is far from ideal. 
We create two models, one to be trained with the standard dataset, and the second, trained with the Gaussian Quantile transformed dataset. 
Below, I plot the input data, and the output data overlapped on each other as histograms. 

1. Standard Model Overlap Model

![alt text](https://github.com/VANRao-Stack/gsoc_eval_task/blob/main/GSoC/standard.png)

2. Gauss Model Overlap Model

![alt text](https://github.com/VANRao-Stack/gsoc_eval_task/blob/main/GSoC/gauss.png)

We see that both the models have learned a very decent representation of the given data, and with a training time of just around 200 epochs (around 2 minutes on a CPU), the model performs quite well. 

# Testing
We now use the test dataset prepared at the start to see how the two models perform on data it has not seen. 
We see that the model trained on the standard dataset has an average MSE loss of the order of 10**(-6), whereas the model trained on the Gaussian Quantile transformed data has an average MSE loss of the order 10**(-5). 
Note that the autoencoder selected was selected based on Eric Wulff's thesis, which was based on the standard dataset, being so close in average MSE loss, it would be interesting to see how another model of a different size designed to the Gaussian Quantile transformed data, would fair against the original model. 

