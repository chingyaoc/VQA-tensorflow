# Tensorflow Implementation of Deeper LSTM+ normalized CNN for Visual Question Answering

![](https://cloud.githubusercontent.com/assets/19935904/16358326/e6812310-3add-11e6-914f-c61c19d6ab5a.png)

Provide tensorflow edition for VQA_LSTM_CNN, training a deeper LSTM and normalized CNN Visual Question Answering model. The current code can reach same accuracy with original torch code on Open-Ended (use COCO train set to train and validation set to evaluation). You can check original torch edtion from [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN) for more details.

### Requirements

This code is written in Python and requires [Tensorflow](https://www.tensorflow.org). The preprocssinng code is in Python.

### Prepare Data (from [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN))
(Here's a copy from the original readme.md)
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run

```
$ python vqa_preprocessing.py --download True --split 1
```

`--download Ture` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) and `--split 1` means you use COCO train set to train and validation set to evaluation. `--split 2 ` means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

Once you have these, we are ready to get the question and image features. Back to the main folder, run

```
$ python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
```

to get the question features. `--num_ans` specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in your main folder, `data_prepro.h5` and `data_prepro.json`. To get the image features, run

```
$ th prepro_img.lua -input_json data_prepro.json -image_root path_to_image_root -cnn_proto path_to_cnn_prototxt -cnn_model path to cnn_model
```

Here we use VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77). After this step, you can get the image feature `data_img.h5`. We have prepared everything and ready to launch training. You can simply run


### Training and Testing

To train on the prepared dataset, comment out `test()`.
We simply run the program with python.

```
$ python model_VQA.py
```

with the default parameter, this will take several hours and will generate the model under `model_save`
To test, comment out `train()` and run the same program, this will generate `data.json`
We finally run a simple program to correct the generated json files.

```
$ python s2i.py
```

This will generate the result `OpenEnded_mscoco_lstm_results.json`. To evaluate the accuracy of generate result, you need to download the [VQA evaluation tools](https://github.com/VT-vision-lab/VQA).

