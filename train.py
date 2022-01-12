#之前使用歐幾里得距離計算相似程度     現在改採餘弦相似度
#這樣減少了計算量 (計算內積只要找到相同的詞語即可  計算距離則是要全部算，但考慮了較多的細節)

#TF-IDF是向量空間模型(VSM) => 可將句子分成詞語，並以向量表示
#TF-IDF很適合拿來擷取關鍵字   但對於QA問答來說  因為它沒有考慮詞語的位置順序，可能還是沒那麼好



import time
import numpy as np
import jieba
import math
import os


def is_all_chinese(string):
    """
    檢查某字串是否全為中文
    
    參數string : 字串
    
    回傳值 : 布林值
    """
    for char in string:
        if not "\u4e00" <= char <= "\u9fa5":
            return False
    return True

def compute_term_frequency(string):
    """
    計算字串(句子)的詞頻

    參數string : 字串
    
    回傳值 : 儲存詞頻的字典(若為空代表字串不包含中文)
    """
    dict_result = {}
    term_count = 0    #中文詞語的數量
    string_segment = jieba.cut(string)  #斷詞
    #計算輸入的詞頻(TF)
    for term in string_segment:
        if is_all_chinese(term):   #只處理中文詞語
            term_count += 1
            if term in dict_result:
                dict_result.update({term : dict_result[term] + 1})
            else:
                dict_result.update({term : 1})
    if term_count == 0:  #沒有任何中文詞語  
        dict_result = {}
    else:
        #將詞頻正規化
        for term in dict_result.keys():
            dict_result.update({term : dict_result[term] / term_count}) 
    return dict_result 

def main():
    time_start = time.time()
    #先獲取目前所在的目錄 (current directory)
    cd = os.getcwd()
    #找到用來存放資料集的目錄
    data_cd = os.path.join(cd, "dataset") 
    #結巴原來的辭典可能會是中國常用的詞，所以要選擇我們自己的台灣常用詞語字典
    #建立台灣常用詞語字典 
    dict_term_TW = {}
    with open(os.path.join(data_cd, "lexicon1_raw_nosil.txt"), encoding="utf8") as file:
        line = file.readline()
        while line:
            line_list = line.split()
            dict_term_TW.update({line_list[0] : 0})   #詞語 : 出現次數
            line = file.readline()
    #將字典存成符合結巴要求的檔案格式
    with open(os.path.join(data_cd, "dict_TW.txt"), 'w', encoding="utf8") as file:
        for term in dict_term_TW:
            file.write(term + " 1 " + "\n")
    jieba.set_dictionary(os.path.join(data_cd, "dict_TW.txt"))     #設定結巴斷詞使用的辭典
    documents_TF = []      #TF表 存放所有檔案中每個詞語出現的次數(term frequency)
    document_count = 0     #總檔案數量
    DF = {}                #DF表 存放每個詞語在多少個檔案中出現
    IDF = {}               #IDF表 即DF的倒數
    documents_TF_IDF = []  #TF-IDF表 (TF-IDF = TF * IDF = TF / DF)
    questions = []          #儲存所有檔案的問題
    answers = []            #儲存所有檔案的回答
    with open(os.path.join(data_cd, "Gossiping-QA-Dataset.txt"), encoding = "utf-8") as file:
        line = file.readline()
        while (line):   #每次處理一行(即一個檔案)
            document_count += 1
            document_TF = {}    #每個question，即每個檔案的詞頻表
            #先分割出question  answer
            line_list = line.split("\t")  #使用tab做切割  
            if len(line_list) == 2:
                #儲存question
                questions.append(line_list[0])
                #儲存answer
                answers.append(line_list[1])
                #計算question的詞頻(TF)
                question = line_list[0]   
                document_TF = compute_term_frequency(question)            
            #計算DF(某詞語在檔案出現的次數) => 若某詞語有在此檔案出現 就加1
            for term in document_TF.keys(): 
                if term in DF:
                    DF.update({term : DF[term] + 1})
                else:
                    DF.update({term : 1})
            #將這個檔案的詞頻加入TF表
            documents_TF.append(document_TF) 
            line = file.readline()
    #依總檔案數量對DF表作正規化、並做平滑化  最後轉換成IDF表
    for term in DF.keys(): 
        #正規化
        #要注意DF有可能出現0的情況(某個詞語完全沒在檔案出現過)  
        #因此要把沒出現過的詞語的DF設為1 => 因此其他有出現過的詞語的DF要再加1
        normalization = (1 + DF[term]) / document_count  #多加1是特殊處理
        #平滑化
        smoothing = math.log10(normalization) #因為檔案數量很大 為了不要讓彼此的差距相差過大 所以要做平滑
        DF.update({term : smoothing})
        IDF.update({term : -smoothing}) #IDF和DF原本是倒數關係 取對數後會差一個負號
    
    #統整TF和IDF表  結合成TF-IDF表    
    for document_TF in documents_TF:
        document_TF_IDF = {}
        length = 0     #某檔案TF-IDF向量的長度(範數)
        for term in document_TF.keys():
            TF_IDF_value = document_TF[term] * IDF[term]
            length += TF_IDF_value * TF_IDF_value  
            document_TF_IDF.update({term : TF_IDF_value})
        document_TF_IDF.update({"document_length" : np.sqrt(length)})  #先儲存這個向量的長度
        documents_TF_IDF.append(document_TF_IDF)
    
    time_end = time.time()
    print("模型訓練花費時間 : %.2f秒" % (time_end - time_start))
    print("\n儲存模型中...")
    
    #以上訊練完模型  要儲存
    time_start = time.time()
    #設定要模型存放的目錄
    model_cd = os.path.join(cd, "model")
    if os.path.isdir(model_cd) == False:   #若無此目錄 就建立
        os.makedirs(model_cd)
    with open(os.path.join(cd, "model", "IDF_model.txt"), 'w', encoding="utf8") as file:
        for term in IDF.keys():
            file.write(term + " " + str(IDF[term]) + "\n")
    with open(os.path.join(cd, "model", "TF_IDF_model.txt"), 'w', encoding="utf8") as file:
        for i in range(len(documents_TF_IDF)):
            document_TF_IDF = documents_TF_IDF[i]
            for term in document_TF_IDF.keys():
                file.write(term + " " + str(document_TF_IDF[term]) + "\n")
            file.write("\n")      
    with open(os.path.join(cd, "model", "QA.txt"), 'w', encoding="utf8") as file:
        for i in range(len(documents_TF_IDF)):
            file.write(questions[i] + "\t" + answers[i])
    time_end = time.time()
    print("儲存模型花費時間 : %.2f秒" % (time_end - time_start))

if __name__ == "__main__": 
    main() 



