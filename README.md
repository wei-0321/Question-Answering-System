# Question-Answering-System
A traditional Chinese question answering system with TF-IDF(Term Frequency - Inverse Document Frequency) algorithm, using the dataset created from this forum named "ptt".
The dataset includes more than 400000 question-answer pairs which are some gossip in taiwan.

# Overview 
(not down yet)

introduction PPT : 

[Question-Answering-System.pptx](https://github.com/wei-0321/Question-Answering-System/files/7361808/Question-Answering-System.pptx)

Below are some demo pictures:

Type a question you want to ask, then the system will compare the similarity between user-question and dataset-question, finally finds top three similar questions and gives you corresponding three answers.



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
(not down yet)

```
(Path)                           (Description)
                     Main folder     
│  │
│  ├ predict.py                  Main program 
│  │
│  ├ train.py                  
│  │
│  ├  dataset
│  │  │
│  │  ├ 
│  │
│  ├  model
│  │  │
│  │  ├ 
│  │
