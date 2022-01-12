# Question-Answering-System
A traditional Chinese question answering system with TF-IDF(Term Frequency - Inverse Document Frequency) algorithm, using the dataset created from this forum named "ptt".
The dataset includes more than 400000 question-answer pairs which are some gossip in taiwan.

# Overview 

introduction PPT : 

[Question-Answering-System.pptx](https://github.com/wei-0321/Question-Answering-System/files/7361808/Question-Answering-System.pptx)

Below are some demo pictures:

flow chart：
![flowchart](https://user-images.githubusercontent.com/71260071/149165926-73193b5d-6e56-4650-9a80-3916bdecdbd9.PNG)

Type a question you want to ask, then the system will compare the similarity between user-question and dataset-question, finally finds top three similar questions and gives you corresponding three answers.

demo：
![image](https://user-images.githubusercontent.com/71260071/149169374-dda71743-f34a-4083-9024-db876556da56.png)


# Requirements 
package:
- numpy
- jieba

# Usage 

1.Open git bash. 

2.Change the diretory where you want to do download this repository.
```
> cd (your directory)
```
3.Clone this repository. 
```
> git clone https://github.com/wei-0321/Question-Answering-System.git
```
4.Change the diretory to this repository.
```
> cd Question-Answering-System
```
5.Execute the program.
```
> python predict.py
```


# Project Structure

```
(Path)                           (Description)
Question-Answering-System        Main folder     
│  │
│  ├ predict.py                  Main program(use pretrained TF-IDF model to predict) 
│  │
│  ├ train.py                    train TF-IDF model
│  │
│  ├  dataset
│  │  │
│  │  ├ lexicon1_raw_nosil.txt    Traditional Chinese vocabulary
│  │  │
│  │  ├ dict_TW.txt               Traditional Chinese vocabulary(jieba format)    
│  │  │
│  │  ├ Gossiping-QA-Dataset.txt  question-answer pairs
│  │
│  ├  model
│  │  │
│  │  ├ IDF_model.txt
│  │  │
│  │  ├ TF_IDF_model.txt
│  │  │
│  │  ├ QA.txt                    question-answer pairs 
│  │
