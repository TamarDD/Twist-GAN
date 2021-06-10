# Twist-GAN
GAN architecture “with a twist”. <br />
The goal of this architecture is inferring the inner-working of a black-box model.

# Goal
The goal is to generate some data that is likely similar to the data used to train black box model. Since there is no way for the model to know for sure, we will use the confidence scores (i.e., classification) produced by the black-box model as indication. 

# Datasets
In this project we used in the following tabular datasets:
* German_credit.arff
* diabetes.arff

# The training process:
1. For each of the two datasets described above, train a RandomForest model. This model is the “black- box”. Use a random dataset split in order to report the performance of the classifier.
2. For each generated sample, the generator will receive two inputs: a vector of random noise Z
and a desired confidence score (a scalar value) C (#1 in Fig 1)
3. The generator will generate a sample and send it to the discriminator (#2 in Fig 1).
4. Instead of a “real” sample (to which we have no access) the discriminator will receive two scalars: a) C (the same one given to the generator); b) Y – the output of the black-box model you trained on the generated sample (#3 in Fig 1).
5. The goal of the discriminator is to determine which of the two values – C or Y – is the true classification produced by the black-box model. The output of the discriminator, denoted by 𝑦". (#4 in Fig 1).

<p align="center">
  <img src="https://github.com/TamarDD/Twist-GAN/blob/main/gan.png" width="500">
</p>

**Figure 1**: The architecture. The generator receives two inputs – Z and C – and produces a sample. This sample is sent to the discriminator, which also receives C (same C as the generator) and the output of the BB model for this sample. The goal of the discriminator is to determine which classification is the “real” one (i.e., produced by the BB model). 
