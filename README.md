# Star Classification using Artificial Neural Networks

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Solution](#solution)
4. [Discussion and Limitation](#discussion-and-limitation)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
Celestial bodies and stars go through a life cycle that begins with their birth within nebulae. Stars based on their mass may enter the main sequence phase when a steady nuclear fusion begins making the star hot and bright. In a later stage, less massive stars become red giants while more massive stars end in dramatic supernova explosions. After a supernova, they leave behind neutron stars or black holes with elements from their core scattered as stardust. This stardust eventually contributes to the formation of new celestial bodies.

A star has different temperatures, luminosity, radius, magnitude, and color during each stage. These properties are helpful for determining the star type. The primary purpose of this project is to demonstrate that stars follow a specific pattern on the Hertzsprung-Russell Diagram, providing valuable insights into celestial bodies' behavior. This classification can assist in understanding the lifecycle and characteristics of stars and is a fundamental problem in astrophysics.

![Hertzsprung-Russell Diagram](image/hertzsprung-russell-diagram.png)

## Problem Statement
The dataset contains 240 rows and 7 columns of data:
- Absolute Temperature (in K)
- Relative Luminosity (L/Lo)
- Relative Radius (R/Ro)
- Absolute Magnitude (Mv)
- Star Color (White, Red, Blue, Yellow, etc.)
- Spectral Class (O, B, A, F, G, K, M)
- Star Type (Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence, SuperGiants, HyperGiants)

The goal of this project is to predict either the Spectral Class or Star Type utilizing the remaining six columns of data. In both instances, the task involves multi-class classification. Star Type is considered a more suitable target given its unknown nature in astronomy. Conversely, the Spectral Class represents an observable or measurable property. Additionally, the Star Type column exhibits a relatively even distribution, contributing to improved accuracy in both the training and testing phases. Both approaches have been completed, and this report will primarily focus on the star-type prediction approach.

Data Source: [Kaggle Star Dataset](https://www.kaggle.com/datasets/deepu1109/star-dataset)

## Solution
The selection of the Artificial Neural Network (ANN) for this problem was optimal given its capability to effectively capture intricate non-linear relationships within the data. With a proven track record in diverse classification tasks, the ANN's inherent ability to learn from data positions it as a compelling choice for predicting star types using the provided dataset.

### Data Structure Management
The preprocessing of the data involved standardizing numerical features and employing one-hot encoding for categorical features. One-hot encoding is particularly crucial for columns like Spectral Class and Star Color where it eliminates the ordinal relationships within categorical features. Additionally, essential techniques such as train-test split were applied, allocating 64% for training, 16% for validation, and 20% for the final testing phase. Data scaling using mean and standard deviation was also implemented to ensure consistency in the data processing pipeline.

### Data Visualization
The correlation plot provides insights into the relationships between different features. Notably, Absolute Magnitude and Star Type exhibit a robust negative linear correlation of -0.96. In contrast, Radius and Temperature display minimal correlation (0.064). The remaining features showcase moderately correlated relationships with values ranging from 0.4 to 0.7.

![Correlation Plot](image/correlation-plot.png)

The 3D plot visually illustrates the interplay between Absolute Magnitude, Temperature, Spectral Class, and Star Type. Each dot in the plot is color-coded to represent the five distinct types of stars. The grouping of dots with the same color signifies that Absolute Magnitude, Temperature, and Spectral Class collectively contribute to effective star-type classification.

![3D Plot](image/3d-plot.png)

### Hyperparameter Tuning
#### Loss Function
Categorical Cross Entropy was selected as the most suitable choice among the three commonly used loss functions: Mean Squared Error, Binary Cross Entropy, and Categorical Cross Entropy.

#### Learning Rate and Optimizer 
Four optimizers were tested to determine the optimal learning rate and number of epochs. Analysis indicated distinct optimal learning rate ranges for each optimizer: SGD=1, RMSprop=0.01, Adam=0.01, and Adadelta=1. Adam emerged as the best choice with the fastest convergence rate, most stable accuracy, and most tolerant of the initial learning rate setting. The best epoch number for Adam is around 300.

![Optimizer Learning Rate](image/optimizer-learning-rate.png)

### Activation Function Hidden Layers and Units 
Three activation functions for hidden layers were compared: Relu, Tanh, and Sigmoid. Tanh emerged as the frontrunner with superior accuracy in both training and testing, displaying a swifter convergence rate.

#### Activation Function Selection for Hidden Layers and Units 
Tanh was selected as the activation function for the hidden layer and unit based on its average accuracy.

#### Activation Function Selection for the Output Layer 
SoftMax was chosen for the output layer, as it is more suitable for multi-class classification tasks and is already incorporated into the loss function.

### Mini-batch
Different mini-batch sizes were explored, with a batch size of 32 demonstrating the highest accuracy (99.15%) with a computational time of 9.5 seconds. However, given the relatively smaller dataset, mini-batch usage was deemed optional, and computational efficiency was prioritized.

![Mini-batch Comparison](image/mini-batch-comparison.png)

### Cross Validation
5-fold cross validation was employed, yielding an impressive 100% accuracy across all five iterations, showcasing the model's robustness and reliability for predicting star types on previously unseen datasets.

![Cross Validation](image/cross-validation.png)

### Final Testing
In the final testing phase, the model achieved a flawless 100% accuracy on the reserved 20% of the dataset. This impeccable performance solidifies the model's effectiveness and reliability in accurately predicting star types.

## Discussion and Limitation
### Hyperparameters tuning strategy
Two strategies were employed in hyperparameter tuning: grid selection and individual selection in a sequential order. Future investigations could explore alternative hyperparameter tuning strategies, such as Random Search and Automated Hyperparameter Tuning Libraries, to further refine and pinpoint the best hyperparameters.

### Concerns about overfitting
Overfitting was addressed by reserving 20% of the dataset for final testing, applying mini-batch training, and implementing cross-validation. For future investigations, incorporating L1 and L2 regularization techniques along with dropout training methods is recommended to further address overfitting concerns.

## Conclusion
This project demonstrates the efficacy of Artificial Neural Networks (ANN) in the classification of stars based on a comprehensive set of astrophysical features. The meticulous selection of hyperparameters and the adoption of various techniques resulted in a robust model with a flawless 100% accuracy on the final testing phase. This optimized model stands as a valuable tool for star classification, poised to contribute meaningfully to the exploration of our vast cosmic landscape.

## References
- Mantovani, R. G., Rossi, A. L. D., Vanschoren, J., Bischl, B., & de Carvalho, A. C. P. L. F. (Year). Effectiveness of Random Search in SVM hyper-parameter tuning. IEEE Electronic Library (IEL) Conference Proceedings, ISSN: 2161-4393, EISSN: 2161-4407, DOI: 10.1109/IJCNN.2015.7280664.
- Anggoro, D. A., & Mukti, S. S. (Year). Performance Comparison of Grid Search and Random Search Methods for Hyperparameter Tuning in Extreme Gradient Boosting Algorithm to Predict Chronic Kidney Failure. International Journal of Intelligent Engineering & Systems.

## Project Files
- [StarType_Prediction.ipynb](/StarType_Prediction.ipynb)
- [Spectrum_Prediction.ipynb](/Spectrum_Prediction.ipynb)
