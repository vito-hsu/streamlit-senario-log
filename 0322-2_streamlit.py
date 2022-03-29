import streamlit as st
from datetime import datetime
from numpy import array, where, unique
from pandas import DataFrame, pivot_table, concat
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gensim.models.word2vec import Word2Vec, LineSentence



def preprocess(data):
    # 測試請跑下面這段 
    # file = open(r'C:\Users\MyUser\Desktop\PeterLogData\2022_0103_03米_UL_Log\ui\2022-01-03\Sceanrio.txt', 'r', encoding="utf-8")
    # file = open(r'C:\Users\MyUser\Desktop\PeterLogData\2022_0102_03米_UnLoader_Log\UI\2022-01-01\Sceanrio.txt', 'r', encoding="utf-8")
    # file = open(r'Sceanrio.txt', 'r', encoding="utf-8")      
    # 記得要加 【, encoding="utf-8"】，以免產生  【UnicodeDecodeError: 'cp950' codec can't decode......】 
    lines               = []                                                                                                      
    file                = data                                           
    for line in file:
        lines.append(line)                                                                                   
    toCombindLineNumber = where([element[0:4]!="2022" for element in lines])[0].tolist()                        # [lines[i] for i in toCombindLineNumber]
    todelete            = (array(toCombindLineNumber)-array(list(range(0,len(toCombindLineNumber))))).tolist()
    for i in todelete:
        lines.pop(i)
    return lines




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data(data):                            
    lines               = preprocess(data=data)
    data                = lines                                                                                     # data[0].split('[')[1].split('] ')[1]
    n                   = len(data)
    info                = [data[i].split('[')[1].split(']')[0] for i in range(n)]                                   # data[550].split(' ')
    datatime            = [data[i].split(' ')[0] + ' ' + data[i].split(' ')[1] for i in range(n)]
    datatime            = [datetime.strptime(datatime[i], '%Y-%m-%d %H:%M:%S,%f') for i in range(n)]                # type(datatime[0])
    df                  = DataFrame({ 'info':info, 'time':datatime })                                               # df['new_time'] = to_datetime(df['time']).dt.time
    df['new_info']      = df['info'].replace(["INFO ", "DEBUG", "WARN ", "ERROR"], [0,1,2,10])                      # df['new_info'].unique()
    df['message1']      = [(data[i].split('[')[1].split('] ')[1].split(' - ')[0]) for i in range(n)]
    df['message2']      = [data[i].split(' - ')[1] for i in range(n)]                                               # [(data[0].split(' - ')[1])]
    # 測試請跑下面這段 
    # data = df
    return df




# upload data
st.title('Scenario Log Analysis')
mode = st.selectbox("Mode", ("Single File", "Multiple Files"))
if mode == "Single File":
    uploaded_file = st.file_uploader("Upload your scenario log file", type='txt')
    if uploaded_file is not None:     
        data_load_state = st.text('Loading data...')                        # Create a text element and let the reader know the data is loading.
        raw_text        = str(uploaded_file.read(),"utf-8").split('\n')     # file_details = {"FileName":uploaded_file.name, "FileType":uploaded_file.type, "FileSize":uploaded_file.size}    ;   st.write(file_details)  
        data            = load_data(data=raw_text)                          # Load 10,000 rows of data into the dataframe.
        data_load_state.text("Done!")                                       # Notify the reader that the data was successfully loaded.    
    else:
        st.write('<span style="color:Red; font-size: 20px; font-weight: bold;">You have not uploaded the appropriate data.</span>', unsafe_allow_html=True)
        st.write('<span style="font-size: 20px; font-weight: bold;">Please upload the data first,</span>'+'<span style="font-size: 20px; font-weight: bold;"> and we will give you the analytical report automatically</span>'+'<span style="color:Red; font-size: 20px; font-weight: bold;"> in the following.</span>', unsafe_allow_html=True)
elif mode == "Multiple Files":
    uploaded_files = st.file_uploader("Upload your scenario log files", type='txt', accept_multiple_files=True)
    if uploaded_files is not None:     
        data_load_state = st.text('Loading data...')                        # Create a text element and let the reader know the data is loading.
        data           = DataFrame({})
        for uploaded_file in uploaded_files:
            raw_text    = str(uploaded_file.read(),"utf-8").split('\n')
            data_split  = load_data(data=raw_text)
            data        = concat([data, data_split])
        data_load_state.text("Done!")                                       # Notify the reader that the data was successfully loaded.    
    else:
        st.write('<span style="color:Red; font-size: 20px; font-weight: bold;">You have not uploaded the appropriate data.</span>', unsafe_allow_html=True)
        st.write('<span style="font-size: 20px; font-weight: bold;">Please upload the data first,</span>'+'<span style="font-size: 20px; font-weight: bold;"> and we will give you the analytical report automatically</span>'+'<span style="color:Red; font-size: 20px; font-weight: bold;"> in the following.</span>', unsafe_allow_html=True)    
        

data = data.sort_values(by=['time'])                                        # ReOrder (for multiple files uploaded)



# EDA (exploratory data analysis)
st.header('EDA (Exploratory Data Analysis)')

# Table
data2 = data.set_index('time')
if st.checkbox('Show Data Table'):
    if st.checkbox('show Only Error'):
        st.subheader('Table (Error)')
        table = data2.loc[data2['info'] == "ERROR"][['info', 'message1', 'message2']]
        st.dataframe(table)
        st.text(f'Total:{table.shape}')
    else:
        st.subheader('Table')
        table = data2[['info', 'message1', 'message2']]
        st.dataframe(table) 
        st.text(f'Total:{table.shape}')






# Pie Chart
st.subheader('Pie Chart')
labels      = 'INFO', 'DEBUG', 'WARN', 'ERROR'
sizes       = list(data['info'].value_counts()[['INFO ', 'DEBUG', 'WARN ', 'ERROR']])
explode     = (0.1, 0.1, 0.1, 0.5)
fig1, ax1   = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax1.axis('equal')
st.pyplot(fig1)                                                 # plt.show()



# Bar Chart
st.subheader('Bar chart - Errors Count in hours')
data3               = data                                      # 令一個新的變數名稱叫做 data3 進一步處理
data3['hour']       = data3['time'].dt.hour
data3['info_type']  = data3['info']
data3               = data3.loc[data3['info_type'] == "ERROR"][['info_type', 'hour']]
table               = pivot_table(data=data3, values='info_type', index=['hour'], aggfunc='count')
st.bar_chart(table)




# Time-Series Plot
st.subheader('Time-Series Plot - Divided into parts')
total_part      = st.number_input('Enter the total parts number:', value=int(8))
part            = st.slider(label = 'which part to see', min_value= 1, max_value= total_part, value= 1)
cut             = list(range(0, len(data), int(len(data)/total_part)))
figure, axis    = plt.subplots(1, 1)
X               = array(data['time'][cut[part-1]:cut[part]])              # type(df['new_time'][0])   ;   type(df['time'][0])
Y               = array(data['new_info'][cut[part-1]:cut[part]])
axis.plot(X,   Y)  
axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
axis.axhline(y=3, c="red", linewidth=2, zorder=0)
st.text(f'Part:{part}/8')
st.pyplot(figure)                                               # plt.show()












# 0323 加入
# 將原始資料進行前處理 !!
st.header('Text Mining')
st.subheader('Parametry Setting')
df = data   ;   text_num = 0
with open(f'scenario_text_loader.txt', 'w', encoding='utf-8') as f:             # 記得要加 encoding='utf-8' 否則會出錯 !!
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

seed        = 666               # 亂數種子
sg_st       = st.radio("What's your algorithm",  ('CBOW', 'Skip-Gram'))         
sg          = 0 if sg_st == 'CBOW' else 1
window_size = st.number_input('Insert the window_size(default = 5)',      value=int(10),  help='周圍詞彙要看多少範圍')                       
vector_size = st.number_input('Insert the vector_size(default = 100)',    value=int(20),  help='轉成向量的維度 ; 詞向量的維度大小，維度太小會無法有效表達詞與詞的關係，維度太大會使關係太稀疏而難以找出規則') 
min_count   = st.number_input('Insert the min_count(default = 5)',        value=int(1),   help='詞頻少於 min_count 之詞彙不會參與訓練')              
workers     = st.number_input('Insert the workers(default = 3)',          value=int(3),   help='訓練的並行數量, workers is the number of "threads" for the training of the model, higher number = faster training.')         
epochs      = st.number_input('Insert the epochs(default = 5)',           value=int(5),   help='訓練的迭代次數')                 
batch_words = st.number_input('Insert the batch_words',                   value=int(2000),help='每次給予多少詞彙量訓練')  
train_data  = LineSentence(f'scenario_text_loader.txt') # type(train_data)  ;   dir(train_data)


# 測試請跑以下
# sg          = 0
# window_size = 10
# vector_size = 20
# min_count   = 1
# workers     = 3
# epochs      = 5
# batch_words = 2000
# train_data  = LineSentence(f'scenario_text_loader.txt')

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
# model.save(f'scenario_text_loader.model')
# model       = Word2Vec.load(f'scenario_text_loader.model')               



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
we_want     = unique(data2.loc[data2['info']=="ERROR"][['message2']])   # 
we_want_new = []
for item in we_want:
    if '\n' in item:
        item = item.replace('\n', '')
    if '\r' in item:
        item = item.replace('\r', '')
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




# import streamlit as st
# import streamlit.components.v1 as stc

# # File Processing Pkgs
# import pandas as pd
# import docx2txt
# from PIL import Image 
# from PyPDF2 import PdfFileReader
# import pdfplumber


# def read_pdf(file):
# 	pdfReader = PdfFileReader(file)
# 	count = pdfReader.numPages
# 	all_page_text = ""
# 	for i in range(count):
# 		page = pdfReader.getPage(i)
# 		all_page_text += page.extractText()

# 	return all_page_text

# def read_pdf_with_pdfplumber(file):
# 	with pdfplumber.open(file) as pdf:
# 	    page = pdf.pages[0]
# 	    return page.extract_text()

# # import fitz  # this is pymupdf

# # def read_pdf_with_fitz(file):
# # 	with fitz.open(file) as doc:
# # 		text = ""
# # 		for page in doc:
# # 			text += page.getText()
# # 		return text 

# # Fxn
# @st.cache
# def load_image(image_file):
# 	img = Image.open(image_file)
# 	return img 



# def main():
# 	st.title("File Upload Tutorial")

# 	menu = ["Home","Dataset","DocumentFiles","About"]
# 	choice = st.sidebar.selectbox("Menu",menu)

# 	if choice == "Home":
# 		st.subheader("Home")
# 		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
# 		if image_file is not None:
		
# 			# To See Details
# 			# st.write(type(image_file))
# 			# st.write(dir(image_file))
# 			file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
# 			st.write(file_details)

# 			img = load_image(image_file)
# 			st.image(img,width=250,height=250)


# 	elif choice == "Dataset":
# 		st.subheader("Dataset")
# 		data_file = st.file_uploader("Upload CSV",type=['csv'])
# 		if st.button("Process"):
# 			if data_file is not None:
# 				file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
# 				st.write(file_details)

# 				df = pd.read_csv(data_file)
# 				st.dataframe(df)

# 	elif choice == "DocumentFiles":
# 		st.subheader("DocumentFiles")
# 		docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
# 		if st.button("Process"):
# 			if docx_file is not None:
# 				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
# 				st.write(file_details)
# 				# Check File Type
# 				if docx_file.type == "text/plain":
# 					# raw_text = docx_file.read() # read as bytes
# 					# st.write(raw_text)
# 					# st.text(raw_text) # fails
# 					st.text(str(docx_file.read(),"utf-8")) # empty
# 					raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
# 					# st.text(raw_text) # Works
# 					st.write(raw_text) # works
# 				elif docx_file.type == "application/pdf":
# 					# raw_text = read_pdf(docx_file)
# 					# st.write(raw_text)
# 					try:
# 						with pdfplumber.open(docx_file) as pdf:
# 						    page = pdf.pages[0]
# 						    st.write(page.extract_text())
# 					except:
# 						st.write("None")
					    
					
# 				elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
# 				# Use the right file processor ( Docx,Docx2Text,etc)
# 					raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
# 					st.write(raw_text)

# 	else:
# 		st.subheader("About")
# 		st.info("Built with Streamlit")
# 		st.info("Jesus Saves @JCharisTech")
# 		st.text("Jesse E.Agbe(JCharis)")



# if __name__ == '__main__':
# 	main()
