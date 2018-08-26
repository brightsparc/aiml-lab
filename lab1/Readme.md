# Lab 1: Introduction to Amazon Sagemaker

In this lab, we will be training a machine learning algorithm on the MNIST dataset using MXNet and Gluon.  We will deploy this model, and test it by drawing some hand written digits in the browser.

## Pre-requisite

You have an active AWS account with credits and permissions to run sagemaker jobs.  We will spin up the following resources:

* `ml.t2.medium` notebook instance  
* `ml.c4.xlarge` for training
* `ml.m4.xlarge` for deployment

## Steps

Following is a list of steps for training and deploying a machine learning model with Amazon Sagemaker.

  1. [Create Notebook Instance](#create-notebook-instance)
  1. [Launch Sample Notebook](#launch-sample-notebook)
  1. [Download dataset](#download-dataset)
  1. [Train model](#train-model)
  1. [Deploy model](#deploy-model)
  2. [Perform Inference](#perform-inference)
  3. [Delete Endpoint](#delete-endpoint)

### 1. Create Notebook Instance

Open Amazon Sagemker in the AWS console in the `us-east-1` (N. Virginia) region, then click **Create notebook instance** button.

![Create notebook instance](screenshots/00-create-notebook-instance.png)

Specify the notebook details

* Name the notebook eg: `student-01`
* For the IAM role select the existing `AmazonSagemaker-ExecutionRole`
* Leave the remaining defaults and click **Create notebook instance**

![Instance settings](screenshots/01-notebook-details.png)

Your notebook will be pending, once it is ready you can click **open** which will redirect to to your new notebook.

![Pending notebook](screenshots/02-pending-notebook.png)

### 2. Launch Sample Notebook

Navigate to the *SageMaker Examples* tab in the Jupyter notebook and expand the *SageMaker Python SDK* to reveal `mxnet_mnist_with_gluon.ipynb` and click **Use** button on the right hand side.

![Minst Gluon](screenshots/03-examples-minst-gluon.png)

Use default filename and click **Create copy** to launch your new notebook.

![Create Copy](screenshots/04-create-copy.png)

### 3. Download dataset

A Jupyter notebook contains markdown content mixed with code with a series of *cells* than can be execute by clicking the Run button, or pressing *shift+enter* keys.

![Run Notebook](screenshots/05-run-notebook.png?classes=shadow)

Let's click through the first few cells to download data and upload it to an S3 bucket within your account.  You will see the bucket name appear as output form the cell eg: `sagemaker-us-east-1-423079281568 `

![Create Copy](screenshots/07-download-data.png?classes=shadow)

### 4. Train Model

We define MXNet estimator for a multi-layer percetron nerual network architecture. 

![Network architecture](screenshots/08-network-architecture.gif)

The `fit` method will train the neural network for 20 `epochs` or training cycles.  For each cycle, the training and validation `loss` will be output to the log.  

![Training progress](screenshots/08-training-progress.png)

The training process should take approximately 5 minutes on a `ml.c4.xlarge` instance.  Once complete you will be able to inspect the job in the Amazon Sagemaker console.

![Training complete](screenshots/09-training-complete.png)

### 4. Deploy Model

With training complete, you can `deploy` your model so that you can predict the most likely digit given a handwritten input.

Run the deploy step and you will get another progress bar which will in the background create a model and an HTTP endpoint.  

![Training complete](screenshots/10-deploy-progress.png)

This should take approximately 5 miunutes for a `ml.m4.xlarge` endpoint.  In the SageMaker console you will see a *Model* has been created.

![Deploy Model](screenshots/11-deploy-model.png)

Along with an *Endpoint* and its configuration.

![Deploy endpoint](screenshots/12-deploy-endpoint.png)

### 5. Perform Inference

With you endpoint deployed, execute the cell that renders the `HTML` widget that allows you to draw an *handwritten digit* and capture the output of this `28x28` pixel in a `data` variable.  Note you can hold down the mouse and draw, or click on the widget for individual pixels.

Once you are happy with your drawing, execute the `predict` function to return a response that should hopefully match your handwritten digit.

![Deploy endpoint](screenshots/13-perform-inference.png)

### 6. Delete Endpoint

Congratulations you have now completed the lab.  Be sure to execute the `delete_endpoint` step to terminate the instance running your ML inference.

![Deploy endpoint](screenshots/14-delete-endpoint.png)




