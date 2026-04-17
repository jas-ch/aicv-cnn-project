[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Xw4JxtAp)

# links
[asl alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) in kaggle  
&nbsp;&nbsp;asl_short: keeps same test data, reduced train data for each letter (39 in each folder), for testing purposes  
[CNN layers size calculation/architecture plan](https://docs.google.com/document/d/107uARKMyD6E3bLRVnlhtI6Ybiw0m8GApL5mvXp_SVo4/edit?usp=sharing)


# changelog
 - (2/27) likely will test different loss/optimizers
 - (3/3) will possibly implement matplotlib--not included in original, but i feel is good to visualize. also added the nn_classify.py (based on transfer learning classify.py) in order to run test sets and compare actual accuracy of checkpointed models
 - (3/4) finished all four models, will run (and run classify after) on GPU while setting up train/val loss plots
 - (3/6) set up epoch loss collection for all models and loss plot comparison in nn_classify, had to hard code epoch as 5. if i have time, i will add train/val plots for each model [to-do]
 - (3/10) fixed loss plot / epoch val loss collection, CLASSIFY FIX IN PROGRESS D:
 - (3/11) fixed classify, unsure about test results (more in comparisons)


# CNN details
 - automatically checks for cuda
 - includes (code) saves based on whether to simply run it ("f" - doesn't save), to save it without running it ("wq", mostly for testing the save code), or to save and run the following code.
 - prints out and saves a file of epoch results
 - saves checkpoints for each epoch [included in transfer]
 - plots validations losses per epoch for each model on the same graph for comparison (also saved)
 - saves final model when finished running
 - all saves in /saves/ (ignored)
 - dataset images are all 200x200x3. so i am currently trying 192 and 96 on first and second CNNs respectively.


## four versions (and their differences)
### version 1
 - [192 x 192 x 3] size (architecture in doc)
 - optimizer Adam w/ lr 0.001
 - loss CrossEntropy

### version 2
 - [96 x 96 x 3] size (architecture in doc)
 - optimizer Adam w/ lr 0.001
 - loss CrossEntropy

### version 3
 - [192 x 192 x 3]
 - optimizer SGD w/ momentum 0.9 and lr 0.01
 - loss CrossEntropy

### version 4
 - [192 x 192 x 3]
 - optimizer AdaGrad w/ lr 0.01
 - loss CrossEntropy


# comparison
This is based on runs on the full dataset. The validation loss plot can be found at [saves/val_loss.jpg](https://github.com/Bishops-25-26-AICV-Per3/cnn-project-jas-ch/blob/main/saves/val_loss.jpg).

**nn1:** fairly low accuracy. hovers from 0.03 - 0.035. this is the same for both training and validation accuracy. both accuracy and loss don't seem to fluctuate much, just hovering around the same values. the validation (and training, to be honest) loss is around 3.36-3.37, and the model doesn't seem to improve much.

**nn2:** very similar to nn1, loss and accuracy for both train and val don't show much change and hover around 0.03(something) for accuracy and 3.36-3.37 for loss.

**nn3:** in this version (change is in the optimizer, SGD) we see much better results! the training accuracy actively increases from the first epoch to the second, where it remains around 0.97-0.99, only dips back down from 0.99 to 0.97 in Epoch 5. same with the loss, it drops a lot from 0.74 in the first epoch to 0.0-0.1 values in the later ones. since Epoch 5 seems to have done worse than Epoch 4, I think it probably checkpointed Epoch 4 (need to comprae test values which I don't have the .txt for yet)

**nn4:** this also shows a lot of improvement. this is also using a different optimizer, but a different optimizer from nn3 as well (AdaGrad). I think this model does the best. loss drops from 3.7 to 0.0(something) values more consistently throughout the epochs, and accuracy goes up from 0.38 to 0.91 from Epoch 1 to Epoch 2, then settles at 0.99 from Epoch 4 and does not dip in Epoch 5. 

The original transfer learning model had a test output prediction accuracy of 100%, although one or two predictions were 99.9999...% when running it on the test set. Regrettably, I did not save per-epoch values in a file, so I need to wait for test output to further compare. However, it still seems transfer learning model does the best, with nn4 (AdaGrad) and nn3 (SDG w/ momentum) as close seconds, or matching it if test prediction does well.

(UPDATE) The val loss plot, once run on the test set, proves this trend. Both nn1 and nn2 show no improvement in loss throughout the epochs and remain at a higher value. nn3 remains fairly low but dips, possibly overfitting with more and more epochs. nn4 shows a dip in loss as more epochs run.

Running the models on the test set (one of each class) shows odd outputs, each model, regardless of loss in validation, predicting nearly only one type of class for all images, with accuracy hovering around the 0.035 mark. I either loaded the checkpoints incorrectly or the models are still not as good as the transfer learning one, despite their ability in training and validation.