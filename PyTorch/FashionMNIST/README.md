# PyTorch - FashionMNIST 
## Deep Learning 

#### 1. Data
Download the FashonMNIST.py file from GitHub (See https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/Mini_Project/FashionMNIST.py). It is
based on the following website https://www.kaggle.com/zalando-research/fashionmnist/ kernels, which you may want to review before proceeding

#### 2. Questions
1. Run the program in Pycharm and investigate the errors. I have manually added some bugs into the sample program. The first step is to get an understanding of the code and fix the
errors. Once you have fixed the errors, investigate and verify its performance.
2. Modify the program to run on the GPU.
3. Find out, if program is set up to perform stochastic gradient descent or mini-batch gradient descent. Explain your answer.
4. Experiment with different numbers of layers and different numbers of neurons (Deep Network vs Shallow Network). While increasing the number of layers in the network,
make sure the total number of weights and biases in the network are roughly the same as the original. Describe how the performance changes as the number of layers increases –
both in terms of training time and performance.
5. Try several training function from the list on this page: http://pytorch.org/docs/master/optim.html. Compare the performance with gradient descent. Hint: Use the
same seed number.
6. Try different transfer functions and do a comparison study between the transfer functions. Explain your results.
7. Write a script to save all gradients with respect to inputs. Save it as CSV file. Calculate the standard deviation of the gradients for each trail and sort the highest 10 std of the
input.
8. Try to use dropout nodes and check if the performance is better or not. Explain the advantages/disadvantages of using dropout nodes.
9. Make a chart and or table to compare the difference in performance between batch, minibatch, and stochastic gradient descent. Calculate the time it takes to train the network for
each method by using the system (sys) clock command in Python.
10. Plot the confusion matrix (as table or a figure) for your predictions and explain each element of the confusion matrix. Additionally, Calculate the accuracy and misclassification
rate.
11. Write a script or function to visualize the network response (output) for an arbitrary input and compare it with the target associated with that input.
12. Bonus: Plot the ROC curve and explain the results. Calculate AUC and explain the results. Note: You need to explain ROC and AUC and the use of these techniques in pattern
recognition. Hint: There is more than one ROC curve in a multi-class problem.
13. Bonus: You can use torchvison capability of PyTorch to visualize the network. Try using tensorboard and look into weight and biases of network and find out how this information
can be useful to improve your results.
