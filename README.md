# DigitClassify

Feedforward neural networks designed to classify hand drawn images of digits (0 - 9).

Each model is trained on the same 60,000 images and tested on the same 10,000 images but uses a different regularization method.


# How to run the models: (4 Total Models: m1, m2, m3, m4)
  * m1 (no regularization)
  * m2 (L2 regularization)
  * m3 (Dropout)
  * m4 (Early stopping)

I have included a makefile with specific commands to run each model.
   * Use the command `make m(#)` where (#) corresponds to each model number (1 - 4). 
   * Running the make command in the src directory will train the model and print out the classification report for the specified model. 
   * The associated .h5 file, classification heatmap and performance plot will be saved to the folder titled ‘output’ within the src directory (./src/output).

Here is an example for m1: 

`make m1`

output: (Classification Report, Performance Plot, and Class HeatMap)

<img width="700" alt="Screen Shot 2023-02-07 at 4 21 34 PM" src="https://user-images.githubusercontent.com/90650832/217368623-1317dd73-800b-42b3-ba51-8bc00796b556.png">

