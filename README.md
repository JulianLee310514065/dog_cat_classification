# Introduction

這是貓狗分類的程式，主要有兩個程式檔，分別為`train.py`與`test.py`，`train.py`用於訓練模型且會輸出一個`cat_dog.pth`，為模型權重檔，`test.py`會讀取此權重檔並做模型驗證。


# Model and Performance Improvement Methods

套用transfer learning並使用`Efficient B0`搭配`ImageNet`之權重，輸出特徵後連接全連接層將輸出類別降為1，且搭配`Sigmoid`作為激活函數，以做二為分類。

使用`adabelief`優化器取代傳統`SGD`與`Adam`。

# DataSet

訓練與驗證資料必須採用下列形式 : 

1. 全部圖片須放於**一個資料夾**中，切勿分資料夾存放
2. 圖片命名須為`cat*.jpg`或是`dog*.jpg`，`*`為任意字串，因標籤辨識採用檔名辨別故需依照此命名法，程式會自動將包含`cat`字串的圖片之標籤設為1，`dog`則為0

# Train Model

可使用此指令訓練模型

```python
python -m train [-batchsize BATCHSIZE] [-modelname MODELNAME] [-epochs EPOCHS] [-lr LR] [-filepath FILEPATH]
```

變數設定如下 : 
* `-batchsize` : 設定批次大小，預設64。
* `-modelname` : 設定transfer learning之套用模型，使用為`timm`函數庫之模型，預設為`efficientnet_b0`。
* `-epochs` : 訓練多少個回合，預設20。
* `-lr` : 優化器之學習率，預設0.0002。
* `-filepath` : 訓練圖片所在資料夾，輸入為字串，預設為`'train'`。
  
# Test Model

可使用此指令驗證模型
```python
python -m test [-filepath FILEPATH] [-batchsize BATCHSIZE] [-modelname MODELNAME]
```

變數設定如下 : 
* `-batchsize` : 設定批次大小，預設64。
* `-modelname` : 設定transfer learning之套用模型，需與訓練時使用相同之模型，預設為`efficientnet_b0`。
* `-filepath` : 驗證圖片所在資料夾，輸入為字串，預設為`'train'`。

驗證會在終端印出準確度等結果，也會畫出混淆矩陣與ROC，如下: 
![cm](https://github.com/JulianLee310514065/dog_cat_classification/blob/main/Figure_1.png)
![roc](https://github.com/JulianLee310514065/dog_cat_classification/blob/main/Figure_2.png)

