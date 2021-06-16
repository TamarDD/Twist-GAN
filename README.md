# Twist-GAN
Implement a GAN architecture “with a twist”. <br />
The goal of this architecture is inferring the inner-working of a black-box model.

# Goal
The goal is to generate some data that is likely similar to the data used to train black box model. Since there is no way for the model to know for sure, we will use the confidence scores (i.e., classification) produced by the black-box model as indication. 

# Datasets
In this project we used in the following tabular datasets:
* German_credit.arff
* diabetes.arff

The data preprocessing is as following:

* Normalize the numeric features.
* Performing one-hot encoder to categorical features.


# The training process
1. For each of the two datasets described above, train a RandomForest model. This model is the “black- box”. Use a random dataset split in order to report the performance of the classifier.
2. For each generated sample, the generator will receive two inputs: a vector of random noise Z
and a desired confidence score (a scalar value) C (#1 in Fig 1)
3. The generator will generate a sample and send it to the discriminator (#2 in Fig 1).
4. Instead of a “real” sample (to which we have no access) the discriminator will receive two scalars: a) C (the same one given to the generator); b) Y – the output of the black-box model you trained on the generated sample (#3 in Fig 1).
5. The goal of the discriminator is to determine which of the two values – C or Y – is the true classification produced by the black-box model. The output of the discriminator, denoted by 𝑦". (#4 in Fig 1).

<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/gan.png" width="500">
  <br>
  Figure 1: The architecture. The generator receives two inputs – Z and C – and produces a sample. This sample is sent to the discriminator, which also receives C (same C as the generator) and the output of the BB model for this sample. The goal of the discriminator is to determine which classification is the “real” one (i.e., produced by the BB model). 
</p>



# Random Forest Performances
We trained Randeom Forest (RF) model on the both datasets. The RF parameters as follow:
* item n estimators = 100
* item max depth = 20
* item random state = 0

The table and figures below present the statistics of Random Forest and confidence score distribution

<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/rf_sts1.png" width="500">
  <br>
  Table 1: Random forest confidence scores statistics on real samples

</p>

<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/hist_dia_rf.png" width="250">
   <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/hist_ger_rf.png" width="250">
  <br>
  Figure 2: Random forest confidence scores distribution on real samples on Diabetes (left) and German (right) datasets
</p>


# Results

In this section we present the Twist GAN model results. The training process lasts 5000 epochs till coverage and batch size is 128. The loss function was binary cross-entropy and shown in Figure 3

<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/diabetes_results.png" width="250">
   <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/german_results.png" width="250">
  <br> Figure 3: Loss of the generator and the discriminator during training process of Diabetes (left) and German (right) datasets
</p>


After our model has converged, we generate 1,000 samples.  The confidence scores of the generated samples are uniformly sampled from [0,1]. Then, we fed the generated samples tothe black-box model. Statistics on the score distributions are shown in Figure 4 and table 2


 <p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/rf_stat2.png" width="500">
  <br>
  Table 2: Random Forest predictions statistics on generated samples
   
</p>



<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/dia_dist.png" width="250">
   <img src="https://github.com/TamarDD/Twist-GAN/blob/main/images/ger_dist.png" width="250">
  <br>
  Figure 4: The score distributions of Random Forest on the fully-trained generative modelgenerated samples. Blue color indicates on generated samples were classified by discriminatoras produced by the black-box model. Diabetes on the left and German on the right.

</p>



# Discussion 
We can notice from the loss graphs shown in Fig 3 that the Generator loss improves when the Discriminator loss degrades (and vice-versa). However, finally the losses of generator and discriminator were converging to similar value.

Furthermore, we observe (Fig 4) that in German dataset, the model was more successful for confidence scores range [0.47,0.57], since there are more generated samples that predicted as "real" by the discriminator. For Diabetes dataset the successful range is more wider (only 3 samples are classified as "fake").

From score distributions of Random Forest, we can conclude that the model may suffer from mode collapse. When RF predict on real samples the distribution is wider than score distributions of generated samples. We want the GAN to produce a wide variety of outputs, but in practice the model focuses on small set of outputs. A possible explanation is that if a generator produces an especially plausible output, the generator may learn to produce only that output. It seems that the generator is always trying to find the one output that seems most plausible to the discriminator. We can see that the generator mainly generate samples the RF predict on ~0.5 Confidence score. We can assume that those sample managed to confuse the discriminator. In other words, the generator learns to map several different input z values to the same output point. 



