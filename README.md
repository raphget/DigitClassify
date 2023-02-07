# DigitClassify

Built four feedforward networks to classify hand drawn images of digits (0 - 9).
Each network using different regularization methods in order to compare classification accuracy.

# How to run the models: (4 Total Models: m1, m2, m3, m4)

I have included a makefile with specific commands to run each model.
   * Use the command `make m(#)` where (#) corresponds to each model number (1 - 4). 
   * Running the make command in the src directory will train the model and print out the classification report for the specified model. 
   * The associated .h5 file, classification heatmap and performance plot will be saved to the folder titled ‘output’ within the src directory (./src/output).

Here is an example for m1: The resulting files:

`src> make m1`

output:
