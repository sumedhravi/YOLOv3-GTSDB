# YOLOv3-GTSDB
Training a YOLOv3 model on the GTSDB dataset to detect and locate traffic signs and classify them based on the types listed in the GTSDB dataset.

## Files
* Compressed GTSDB dataset
* Config files
* *.data*, *.names* and *.obj* files
* Jupyter notebook featuring the complete process of preparing the data, implementing the data pipeline, initializing and training the model and visualizing the object detection process
* YOLO v3 weights trained using the notebook

## How to Use
The notebook can be trained on a local machine, if you have access to a GPU, as well as a Kaggle or Google Colab Kernel.

* You can either clone this repository or download the files to your own machine or execute this notebook on Google Colab. Remember to enable GPU on Colab.
* Download the dataset from [here](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip)
* Define the values of these flags based on the operations you would like to perform.
* Execute each cell in the same order as in the notebook.
```
# Flags 
train_individual_classes = False # To train for 43 classes instead of the 4 main parent classes
train = True # If the model needs to be trained
load_saved_weights = True #To load the best weights from the previously trained model.
```
To train or use the model to classify all the 43 individual classes instead of the main parent classes (prohibitory, mandatory, danger and other) set ``train_indivdual_classes = True.``

To carry out the training process, set train = True. If you just want to check predictions using a saved model, you can set the flag `train = False.`

If you would like to train the model from scratch using the pretrained weights from [here](https://pjreddie.com/media/files/yolov3.weights), set `load_saved_weights = False`, else the weights included in the repository are loaded.

* In the obj.data file, the backup folder is set to a google drive location, since the training was carried out on a Google Colab GPU. Using a google drive folder to save your weights ensures that your weights are still accessible even if the Colab runtime gets terminated due to inactivity. If you are training on a local machine or have chose not to mount your drive, edit the following line in the obj.data file and change it to the location you wish to save your weights to.
```
backup = /content/drive/MyDrive/yolov3/backup
```

* The saved weights which are loaded for more training or detection are the weights included in the repository. If you wish to use the weights you trained, change the following line to the location of your saved weights. weights_43 is if you trained your model on 43 classes and weights_4 for 4 classes.
```
weights_4 = repo_folder + '/yolov3_custom_train_v2_best.weights'
weights_43 = repo_folder + '/yolov3_custom_train_v1_best.weights'
```

## Predictions
To detect traffic signs in images execute this block of code.
```
!./darknet detector test $data_file $cfg_file /content/drive/MyDrive/yolov3/backup/$weights /content/GTSDB/GTSDB_jpg/00018.jpg
imShow('predictions.jpg')
```
In place of the $weights variable, you can substitute the path of the weights trained by you.

If you wish to generate predictions on images not included in the dataset, use the upload method defined to upload images to Google Colab and substitute the path of the image here.

#### Now all you have to do is run the notebook!

### Credits
The model is based on the repository by [AlexeyAB](https://github.com/AlexeyAB/darknet).

[Tutorial on training YOLOv3 model on Google Colab by the AI Guy](https://colab.research.google.com/drive/1Mh2HP_Mfxoao6qNFbhfV3u28tG8jAVGk#scrollTo=k5SYWDPv7qG-)

https://blog.goodaudience.com/part-1-preparing-data-before-training-yolo-v2-and-v3-deepfashion-dataset-3122cd7dd884
