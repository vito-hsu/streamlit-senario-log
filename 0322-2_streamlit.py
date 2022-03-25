import streamlit as st
from datetime import datetime
from numpy import array, where, histogram, unique, random
from pandas import DataFrame, to_datetime, pivot_table
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.title('Senario Log Analysis')


# @st.cache
def load_data():       # nrows = 10000
    filepath    = r'Sceanrio.txt'
    lines       = []                                                                      
    file        = open(filepath, 'r', encoding="utf-8")        # 記得要加 【, encoding="utf-8"】，以免產生  【UnicodeDecodeError: 'cp950' codec can't decode......】 

    for line in file:
        lines.append(line)      
                                                
                                                
    toCombindLineNumber = where([element[0:4]!="2022" for element in lines])[0].tolist()

    [lines[i] for i in toCombindLineNumber]

    todelete = (array(toCombindLineNumber)-array(list(range(0,len(toCombindLineNumber))))).tolist()
    for i in todelete:
        lines.pop(i)

    data                = lines                                                                                     # data[0].split('[')[1].split('] ')[1]
    info                = [data[i].split('[')[1].split(']')[0] for i in range(len(data))]                           # data[550].split(' ')
    datatime            = [data[i].split(' ')[0] + ' ' + data[i].split(' ')[1] for i in range(len(data))]
    datatime            = [datetime.strptime(datatime[i], '%Y-%m-%d %H:%M:%S,%f') for i in range(len(datatime))]    # type(datatime[0])
    df                  = DataFrame({ 'info':info, 'time':datatime })
    df['new_time']      = to_datetime(df['time']).dt.time

    df['new_info']      = df['info'].replace(["INFO ", "DEBUG", "WARN ", "ERROR"], [0,1,2,10])
    df['error_message1'] = [(data[i].split('[')[1].split('] ')[1].split(' - ')[0]) if df['new_info'][i] == 10 else "" for i in range(len(data))]
    df['error_message2'] = [(data[i].split('[')[1].split('] ')[1].split(' - ')[1]) if df['new_info'][i] == 10 else "" for i in range(len(data))]
    df['message1']      = [(data[i].split('[')[1].split('] ')[1].split(' - ')[0]) for i in range(len(data))]
    df['message2']      = [data[i].split(' - ')[1] for i in range(len(data))]                                       # [(data[0].split(' - ')[1])]
    df['new_info'].unique()

    return df


data_load_state = st.text('Loading data...')        # Create a text element and let the reader know the data is loading.
data = load_data()                                  # Load 10,000 rows of data into the dataframe.
data_load_state.text("Done! (using st.cache)")      # Notify the reader that the data was successfully loaded.


# EDA (exploratory data analysis)
st.header('EDA (exploratory data analysis)')
if st.checkbox('Show data'):
    if st.checkbox('Show error data'):
        st.subheader('Raw error data')
        data2 = data.set_index('time')
        st.dataframe(data2.loc[data2['info'] == "ERROR"].iloc[:,[0,3,4]])
    else:
        st.subheader('Raw data')
        st.dataframe(data.iloc[:,[0,1,6,7]])        





# Pie chart
st.subheader('Info/Debug/Warn/Error Pie Chart')
labels = 'Info', 'Debug', 'Warn', 'Error'
sizes = list(data['info'].value_counts())
explode = (0.1, 0.1, 0.1, 0.5)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax1.axis('equal')
st.pyplot(fig1)



# Bar chart
st.subheader('Errors Count in hours')
data3           = data
data3['hour']   = data3['time'].dt.hour
data3           = data3.loc[data3['info'] == "ERROR"].iloc[:,[0,8]]
table           = pivot_table(data=data3, values='info', index=['hour'], aggfunc='count')
st.bar_chart(table)




# 
df = data
cut1 = int(round(len(df)/8))
cut2 = int(round(len(df)/8*2))
cut3 = int(round(len(df)/8*3))
cut4 = int(round(len(df)/8*4))
cut5 = int(round(len(df)/8*5))
cut6 = int(round(len(df)/8*6))
cut7 = int(round(len(df)/8*7))
cut  = [0, cut1, cut2, cut3, cut4, cut5, cut6, cut7, len(df)]   ;   part = 1

for j in range(8):
    figure, axis = plt.subplots(1, 1)
    X  = array(df['time'][cut[j]:cut[j+1]])             # type(df['new_time'][0])   ;   type(df['time'][0])
    Y  = array(df['new_info'][cut[j]:cut[j+1]])
    axis.plot(X,   Y)  
    ax = axis                                           # ax.set_title('Manual DateFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.axhline(y=3, c="red", linewidth=2, zorder=0)
    st.subheader(f'Part:{part}/8')    ;   part += 1
    st.pyplot(figure)                                   # plt.show()












# 0323 加入
# 將原始資料進行前處理 !!
st.header('Text Mining')
st.subheader('Parametry Setting')
df = data   ;   text_num = 0
with open(f'senario_text_loader.txt', 'w', encoding='utf-8') as f:             # 記得要加 encoding='utf-8' 否則會出錯 !!
    for text in list(df['message2']):                                      
        text = text.replace(' ', '_')
        f.write(''.join(text)+' ')                                       
        text_num += 1
        # 以下僅作為提醒使用!!!
        if text_num % 500 == 0:
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))
            print('{} articles processed.'.format(text_num))


    print('{} articles processed.'.format(text_num))

# 訓練資料 !!
from gensim.models.word2vec import Word2Vec, LineSentence
seed        = 666               # 亂數種子

sg_st       = st.radio("What's your algorithm",  ('CBOW', 'Skip-Gram'))         
sg          = 0 if sg_st == 'CBOW' else 1

window_size = st.number_input('Insert the window_size(default = 5)',      value=int(10),  help='周圍詞彙要看多少範圍')                       
vector_size = st.number_input('Insert the vector_size(default = 100)',    value=int(20),  help='轉成向量的維度 ; 詞向量的維度大小，維度太小會無法有效表達詞與詞的關係，維度太大會使關係太稀疏而難以找出規則') 
min_count   = st.number_input('Insert the min_count(default = 5)',        value=int(1),   help='詞頻少於 min_count 之詞彙不會參與訓練')              
workers     = st.number_input('Insert the workers(default = 3)',          value=int(3),   help='訓練的並行數量, workers is the number of "threads" for the training of the model, higher number = faster training.')         
epochs      = st.number_input('Insert the epochs(default = 5)',           value=int(5),   help='訓練的迭代次數')                 
batch_words = st.number_input('Insert the batch_words',                   value=int(2000),help='每次給予多少詞彙量訓練')  
train_data  = LineSentence(f'senario_text_loader.txt') # type(train_data)  ;   dir(train_data)

model       = Word2Vec(
    train_data,
    min_count   =   min_count,
    vector_size =   vector_size,
    workers     =   workers,
    epochs      =   epochs,
    window      =   window_size,
    sg          =   sg,
    seed        =   seed,
    batch_words =   batch_words
)
# model.save(f'senario_text_loader.model')
# model       = Word2Vec.load(f'senario_text_loader.model')               



# 模型測試
# Results1
st.subheader('Results1')
words1      = st.text_input('Enter the words', '板件讀取Unknown寫入失敗！currentSlotLocation:0、panelDataList.Count:24')                   # words1 = '尚未讀板，無法下發 Move Out 指令，請檢查步序是否正確。'
result_df1  = DataFrame({})
for item in model.wv.most_similar(words1):                              # item[0]
    new_row     = {'text':item[0], 'prob':item[1]}
    result_df1  = result_df1.append(new_row, ignore_index = True)       # result_df1['text'][0]
st.write('The most similar words to the words you enter:', result_df1)


# Results2
st.subheader('Results2')
data2       = data.set_index('time') 
we_want     = unique(data2.loc[data2['info'] == "ERROR"].iloc[:,4])
we_want_new = []
for item in we_want:
    if '\n' in item:
        item = item.replace('\n', '')
    if ' ' in item:
        item = item.replace(' ', '_')
    we_want_new.append(item)
        
words2      = st.selectbox('Select the words (Error Messages)', we_want_new)               # unique(data2.loc[data2['info'] == "ERROR"].iloc[:,4])              
result_df2  = DataFrame({})
for item in model.wv.most_similar(words2):                            
    new_row     = {'text':item[0], 'prob':item[1]}
    result_df2  = result_df2.append(new_row, ignore_index = True)
st.write('The most similar words to the words you select:', result_df2)






# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data



# data_load_state = st.text('Loading data...')        # Create a text element and let the reader know the data is loading.
# data = load_data(10000)                             # Load 10,000 rows of data into the dataframe.
# data_load_state.text("Done! (using st.cache)")      # Notify the reader that the data was successfully loaded.


# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)



# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)


# # Some number in the range 0-23
# hour_to_filter = st.slider(label= 'hour', min_value= 0, max_value= 23, value= 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]


# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)




