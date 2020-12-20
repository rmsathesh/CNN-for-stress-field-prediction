# CNN-for-stress-field-prediction

This project is developed to predict the stress field in solid material elastic deformation using convolutional neural network. A Squeeze and Excitation residual blocks embedded fully convolutional neural network with multiple input channels (consisting of geometry, loads & boundary conditions) was developed to accurately predict the stress field. Proposed model has better computational efficiency than FEA model, so this can be used as an alternative during structural design and topology optimization.

Below paper was taken as a reference for this model.
'Nie, Zhenguo & Jiang, Haoliang & Kara, Levent. (2018). Deep Learning for Stress Field Prediction Using Convolutional Neural Networks.'

Model architecture was fine tuned a bit to reduce computational load. Results showed good correlation with the ground truth value.

Input image consists of 5 channels.

Channel-1: Gemetric information of the beam. This is binary encoded. 1 represents material is present and 0 represents no material.
Channel-2: Information regarding x component of force. Magnitude of force is encoded on the respective pixel where force is applied.
Channel-3: Information regarding y component of force. Magnitude of force is encoded on the respective pixel where force is applied.
Channel-4: Information regarding x component of boundary condition. -1 where the beam is fixed and 0 otherwise
Channel-5: Information regarding y component of boundary condition. -1 where the beam is fixed and 0 otherwise

A simple case of cantilever beam fixed at one end and a uniform force applied at the opposite end is considered for our problem. Cantilever beam of various shapes and different magnitude of forces were taken as training set. 

Results showed good accuracy with FEA model. 



