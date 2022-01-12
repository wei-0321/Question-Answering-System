#web socket有許多優點(業界許多工程師之間的資料傳遞或測試都是靠這種API) 
#做成web socket的模式     盡量不要採用本地端的形式(限制很多)



import time
import numpy as np
import jieba
import math
import os
from train import is_all_chinese, compute_term_frequency

def remove_space(string):
    """
    去除字串中的所有空白字元
    
    參數string : 字串
    
    回傳值 :去掉所有空白字元的字串
    """
    result = ""
    for char in string:
        if char != ' ':
            result += char
    return result

def compute_distance(dict_TF1, dict_TF2):  
    """
    計算兩字典的歐幾里得距離
    
    參數dict_TF1 : TF-IDF表(字典)
    參數dict_TF2 : TF-IDF表(字典)
    
    回傳值 : 歐幾里得距離
    
    若有先儲存向量長度 不能把它列入距離的計算
    跟計算內積相比  運算量較多
    """
    diff_sum = 0
    dict_TF2_temp = dict_TF2.copy()   #複製 避免參照的問題
    for key in dict_TF1.keys():
        if key in dict_TF2:           #兩者共同的鍵
            diff_sum += np.power(dict_TF1[key] - dict_TF2_temp[key], 2)
            dict_TF2_temp.pop(key, 0)  #移除已經算過的鍵
        else:
            diff_sum += np.power(dict_TF1[key] - 0, 2)
    for key in dict_TF2_temp.keys():     #剩下的鍵都不是共同的
        diff_sum += np.power(dict_TF2_temp[key] - 0, 2)
    return np.sqrt(diff_sum)

def compute_cos_similarity(dict_TF1, dict_TF2): 
    """
    計算兩字典的餘弦相似度
    
    參數dict_TF1 : 問句的TF-IDF表(字典)
    參數dict_TF2 : 某檔案的TF-IDF表(字典)
    
    回傳值 : 餘弦相似度數值
    """
    dot = 0  #內積值
    denominator = dict_TF1["query_length"] * dict_TF2["document_length"]   #分母(兩向量的長度相乘)
    if denominator == 0:
        denominator = 1
    for key in dict_TF1.keys():
        if key in dict_TF2:           #內積的計算只要找兩者共同的鍵(詞語)
           dot += dict_TF1[key] * dict_TF2[key]
    cos_similarity = dot / denominator
    return cos_similarity

def compare(top_three, document_index, similarity):
    """
    維護一個字典，使字典存放的是與問題最像的三個檔案
    
    參數top_three : 存放目前相似度前三名的字典
    參數document_index : 位於TF-IDF表中  某個檔案的索引值
    參數similarity : 某檔案與問題的餘弦相似度
    """
    top_three.update({document_index : similarity})
    if len(top_three) > 3:
        pop_key = document_index
        min_value = similarity   #cos值介於-1 ~ 1，兩向量相差0度角為最像(一樣)，因此只要找最大值即可
        for key in top_three.keys():
            if top_three[key] < min_value:
                pop_key = key
                min_value = top_three[key]
        top_three.pop(pop_key) 
        
def query(sentence, documents_TF_IDF, IDF, document_count):
    """
    將使用者的問題與模型中的檔案比對，找出最相似的作為答案
    
    參數sentence : 使用者輸入的問題句子(字串)
    參數documents_TF_IDF : TF-IDF表(串列，裡面的元素為字典)
    參數IDF : IDF表(字典)
    參數document_count : 總檔案(QA)數量
    
    回傳值 : 存放前三個最相似檔案的字典 => 鍵:檔案索引值 / 值:距離  
    (若使用者輸入形式有誤  會有防呆訊息)
    """
    top_three = {}   #三個與問題最像的檔案
    #計算輸入的詞頻(TF)
    query_TF = compute_term_frequency(sentence)
    #檢查輸入
    if len(query_TF) == 0:   #使用者的輸入沒有包含中文字
        return "請輸入至少一個中文字!"    #防呆訊息
    #若輸入沒問題  開始比較每個檔案的TF-IDF並找出最相似的三個檔案(即餘弦相似度值要跟1最近) 
    else:
        #要先將輸入的TF轉換為TF-IDF
        query_TF_IDF = {}
        length = 0     #問題TF-IDF向量的長度
        for term in query_TF.keys():
            TF_IDF_value = 0
            if term not in IDF:
                #沒在檔案出現過的詞語預設成1
                TF_IDF_value = query_TF[term] * math.log10(document_count / 1)
            else:      
                TF_IDF_value = query_TF[term] * IDF[term]
            query_TF_IDF.update({term : TF_IDF_value})
            length += TF_IDF_value * TF_IDF_value
        query_TF_IDF.update({"query_length" : np.sqrt(length)})
        print(query_TF_IDF)
        #找出最相似的檔案
        for i in range(len(documents_TF_IDF)):
            #依序取出各個檔案
            document_TF_IDF = documents_TF_IDF[i]
            #計算歐幾里得距離
            #distance = compute_distance(query_TF_IDF, document_TF_IDF)           
            #計算餘弦相似度
            similarity = compute_cos_similarity(query_TF_IDF, document_TF_IDF)
            #維護與問題最相似的檔案之字典(鍵:檔案索引值 / 值:相似度)
            compare(top_three, i, similarity)
        return top_three   #回傳存放前三個最相似檔案的字典
    
def main():    
    time_start = time.time()
    #先獲取目前所在的目錄 (current directory)
    cd = os.getcwd()
    #找到用來存放模型的目錄
    model_cd = os.path.join(cd, "model")  
    #設定結巴斷詞使用的辭典
    jieba.set_dictionary(os.path.join(cd, "dataset","dict_TW.txt"))     
    #載入模型
    questions = []
    answers = []
    document_count = 0     #總檔案數量
    IDF = {}               #IDF表 
    documents_TF_IDF = []  #TF-IDF表 
    with open(os.path.join(model_cd, "IDF_model.txt"), 'r', encoding="utf8") as file:
        line = file.readline()
        while (line):   #每次處理一行(即一個檔案)
            line_list = line.split(" ")
            IDF.update({line_list[0] : float(line_list[1])})
            line = file.readline()
    with open(os.path.join(model_cd, "TF_IDF_model.txt"), 'r', encoding="utf8") as file:
        line = file.readline()
        document_TF_IDF = {}
        while (line):   #每次處理一行(即一個檔案)
            line_list = line.split(" ")
            if len(line_list) == 2:
                document_TF_IDF.update({line_list[0] : float(line_list[1])})
            else:
                documents_TF_IDF.append(document_TF_IDF)
                document_TF_IDF = {}
            line = file.readline()
    with open(os.path.join(model_cd, "QA.txt"), 'r', encoding="utf8") as file:
        line = file.readline()
        while (line):   #每次處理一行(即一個檔案)
            document_count += 1
            #先分割出question  answer
            line_list = line.split("\t")  #使用tab做切割 
            questions.append(line_list[0])
            answers.append(line_list[1])
            line = file.readline()
    time_end = time.time()
    print("載入模型花費時間 : %.2f秒" % (time_end - time_start))
    #預測
    q = remove_space(input("請輸入問題 : "))
    while len(q) > 0:
        time_start = time.time()
        dict_reply = query(q, documents_TF_IDF, IDF, document_count)
        #將字典依照相似度高低作排序
        dict_reply = dict(sorted(dict_reply.items(), key=lambda x:x[1], reverse = True))
        for key in dict_reply.keys():
            print("\n相似問題 : ", questions[key])
            print("TF-IDF : ", documents_TF_IDF[key])
            #將餘弦相似度的值(-1 ~ 1)對應到(0 ~ 100)
            print("相似度 :　" + str(round(dict_reply[key] * 50 + 50, 3)) + "%")
            print("回答 : ", answers[key])
        time_end = time.time()
        print("回答時間 : %.2f秒" % (time_end - time_start))
        q = remove_space(input("\n(若要停止 請直接按下ENTER鍵)\n 請輸入問題 : "))
    
if __name__ == "__main__": 
    main() 