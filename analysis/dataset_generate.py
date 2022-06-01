import scipy, pandas, numpy
import matplotlib.pyplot as plt  

def multichoice(df, col_key):
    df_copy  = df.copy()
    col=[]
    if isinstance(col_key,list):
        for i in col_key:
            col += df.filter(like=i).columns.tolist()
        # print(col)
    else:
        col = df.filter(like=col_key).columns.tolist()
    df[col] = df_copy[col].replace(-99.99, 0.00,inplace=False)
    df.loc[(df[col]==0.00).all(axis=1), col] = numpy.nan
    return df

df = pandas.read_csv('sme_finance_monitor_q2_2018-q3_2020v2.csv')

"""Process the outcome of data"""
df_new = df
df_new['outcome'] = -1

df_new.loc[((df_new['q57_1']>0)&(df_new['q57_1']<5))|((df_new['q57_2']>0)&(df_new['q57_2']<5))|
            ((df_new['q57_3']>0)&(df_new['q57_3']<5))|((df_new['q57_4']>0)&(df_new['q57_4']<5))|
            ((df_new['q57_7']>0)&(df_new['q57_7']<5))|((df_new['q57_8']>0)&(df_new['q57_8']<5))|
            ((df_new['q57_9']>0)&(df_new['q57_9']<5))|((df_new['q57_10']>0)&(df_new['q57_10']<5))|
            ((df_new['q57_11']>0)&(df_new['q57_11']<5))|((df_new['q57_12']>0)&(df_new['q57_12']<5))|
            ((df_new['q57_13']>0)&(df_new['q57_13']<5))|((df_new['q57_14']>0)&(df_new['q57_14']<5))|
            ((df_new['q57_15']>0)&(df_new['q57_15']<5))|((df_new['q57_16']>0)&(df_new['q57_16']<5))|
            ((df_new['q39_1_1']>0)&(df_new['q39_1_1']<5))|((df_new['q39_2_1']>0)&(df_new['q39_2_1']<5))|
            ((df_new['q39_3_1']>0)&(df_new['q39_3_1']<5))|((df_new['q39_4_1']>0)&(df_new['q39_4_1']<5))|
            ((df_new['q39_1_2']>0)&(df_new['q39_1_2']<5))|((df_new['q39_2_2']>0)&(df_new['q39_2_2']<5))|
            ((df_new['q39_3_2']>0)&(df_new['q39_3_2']<5))|((df_new['q39_4_2']>0)&(df_new['q39_4_2']<5))|
            ((df_new['q39_1_3']>0)&(df_new['q39_1_3']<5))|((df_new['q39_2_3']>0)&(df_new['q39_2_3']<5))|
            ((df_new['q39_3_3']>0)&(df_new['q39_3_3']<5))|((df_new['q39_4_3']>0)&(df_new['q39_4_3']<5))|
            ((df_new['q39_1_4']>0)&(df_new['q39_1_4']<5))|((df_new['q39_2_4']>0)&(df_new['q39_2_4']<5))|
            ((df_new['q39_3_4']>0)&(df_new['q39_3_4']<5))|((df_new['q39_4_4']>0)&(df_new['q39_4_4']<5))|
            ((df_new['q39_1_7']>0)&(df_new['q39_1_7']<5))|((df_new['q39_2_7']>0)&(df_new['q39_2_7']<5))|
            ((df_new['q39_3_7']>0)&(df_new['q39_3_7']<5))|((df_new['q39_4_7']>0)&(df_new['q39_4_7']<5))|
            ((df_new['q39_1_8']>0)&(df_new['q39_1_8']<5))|((df_new['q39_2_8']>0)&(df_new['q39_2_8']<5))|
            ((df_new['q39_3_8']>0)&(df_new['q39_3_8']<5))|((df_new['q39_4_8']>0)&(df_new['q39_4_8']<5))|
            ((df_new['q39_1_9']>0)&(df_new['q39_1_9']<5))|((df_new['q39_2_9']>0)&(df_new['q39_2_9']<5))|
            ((df_new['q39_3_9']>0)&(df_new['q39_3_9']<5))|((df_new['q39_4_9']>0)&(df_new['q39_4_9']<5))|
            ((df_new['q39_1_10']>0)&(df_new['q39_1_10']<5))|((df_new['q39_2_10']>0)&(df_new['q39_2_10']<5))|
            ((df_new['q39_3_10']>0)&(df_new['q39_3_10']<5))|((df_new['q39_4_10']>0)&(df_new['q39_4_10']<5))|
            ((df_new['q39_1_11']>0)&(df_new['q39_1_11']<5))|((df_new['q39_2_11']>0)&(df_new['q39_2_11']<5))|
            ((df_new['q39_3_11']>0)&(df_new['q39_3_11']<5))|((df_new['q39_4_11']>0)&(df_new['q39_4_11']<5))|
            ((df_new['q39_1_12']>0)&(df_new['q39_1_12']<5))|((df_new['q39_2_12']>0)&(df_new['q39_2_12']<5))|
            ((df_new['q39_3_12']>0)&(df_new['q39_3_12']<5))|((df_new['q39_4_12']>0)&(df_new['q39_4_12']<5))|
            ((df_new['q39_1_13']>0)&(df_new['q39_1_13']<5))|((df_new['q39_2_13']>0)&(df_new['q39_2_13']<5))|
            ((df_new['q39_3_13']>0)&(df_new['q39_3_13']<5))|((df_new['q39_4_13']>0)&(df_new['q39_4_13']<5))|
            ((df_new['q39_1_14']>0)&(df_new['q39_1_14']<5))|((df_new['q39_2_14']>0)&(df_new['q39_2_14']<5))|
            ((df_new['q39_3_14']>0)&(df_new['q39_3_14']<5))|((df_new['q39_4_14']>0)&(df_new['q39_4_14']<5))|
            ((df_new['q39_1_15']>0)&(df_new['q39_1_15']<5))|((df_new['q39_2_15']>0)&(df_new['q39_2_15']<5))|
            ((df_new['q39_3_15']>0)&(df_new['q39_3_15']<5))|((df_new['q39_4_15']>0)&(df_new['q39_4_15']<5))|
            ((df_new['q39_1_16']>0)&(df_new['q39_1_16']<5))|((df_new['q39_2_16']>0)&(df_new['q39_2_16']<5))|
            ((df_new['q39_3_16']>0)&(df_new['q39_3_16']<5))|((df_new['q39_4_16']>0)&(df_new['q39_4_16']<5))|
            ((df_new['q39_1_18']>0)&(df_new['q39_1_18']<5))|((df_new['q39_2_18']>0)&(df_new['q39_2_18']<5))|
            ((df_new['q39_3_18']>0)&(df_new['q39_3_18']<5))|((df_new['q39_4_18']>0)&(df_new['q39_4_18']<5))
            ,'outcome'] = 1

df_new.loc[(df_new['q57_1']==5)|(df_new['q57_2']==5)|(df_new['q57_3']==5)|
            (df_new['q57_4']==5)|(df_new['q57_7']==5)|(df_new['q57_8']==5)|
            (df_new['q57_9']==5)|(df_new['q57_10']==5)|(df_new['q57_11']==5)|
            (df_new['q57_12']==5)|(df_new['q57_13']==5)|(df_new['q57_14']==5)|
            (df_new['q57_15']==5)|(df_new['q57_16']==5)|
            (df_new['q39_1_1']==5)|(df_new['q39_2_1']==5)|(df_new['q39_3_1']==5)|(df_new['q39_4_1']==5)|
            (df_new['q39_1_2']==5)|(df_new['q39_2_2']==5)|(df_new['q39_3_2']==5)|(df_new['q39_4_2']==5)|
            (df_new['q39_1_3']==5)|(df_new['q39_2_3']==5)|(df_new['q39_3_3']==5)|(df_new['q39_4_3']==5)|
            (df_new['q39_1_4']==5)|(df_new['q39_2_4']==5)|(df_new['q39_3_4']==5)|(df_new['q39_4_4']==5)|
            (df_new['q39_1_7']==5)|(df_new['q39_2_7']==5)|(df_new['q39_3_7']==5)|(df_new['q39_4_7']==5)|
            (df_new['q39_1_8']==5)|(df_new['q39_2_8']==5)|(df_new['q39_3_8']==5)|(df_new['q39_4_8']==5)|
            (df_new['q39_1_9']==5)|(df_new['q39_2_9']==5)|(df_new['q39_3_9']==5)|(df_new['q39_4_9']==5)|
            (df_new['q39_1_10']==5)|(df_new['q39_2_10']==5)|(df_new['q39_3_10']==5)|(df_new['q39_4_10']==5)|
            (df_new['q39_1_11']==5)|(df_new['q39_2_11']==5)|(df_new['q39_3_11']==5)|(df_new['q39_4_11']==5)|
            (df_new['q39_1_12']==5)|(df_new['q39_2_12']==5)|(df_new['q39_3_12']==5)|(df_new['q39_4_12']==5)|
            (df_new['q39_1_13']==5)|(df_new['q39_2_13']==5)|(df_new['q39_3_13']==5)|(df_new['q39_4_13']==5)|
            (df_new['q39_1_14']==5)|(df_new['q39_2_14']==5)|(df_new['q39_3_14']==5)|(df_new['q39_4_14']==5)|
            (df_new['q39_1_15']==5)|(df_new['q39_2_15']==5)|(df_new['q39_3_15']==5)|(df_new['q39_4_15']==5)|
            (df_new['q39_1_16']==5)|(df_new['q39_2_16']==5)|(df_new['q39_3_16']==5)|(df_new['q39_4_16']==5)|
            (df_new['q39_1_18']==5)|(df_new['q39_2_18']==5)|(df_new['q39_3_18']==5)|(df_new['q39_4_18']==5)
,'outcome'] = 0

df_new=df_new[~df_new['outcome'].isin([-1])]

df_final = df_new[['outcome','risk','q126','q144','q7q8','q9','q11','q11a','q12','q13','q13a','q13b','q14a','q14y','q14ysu2',
'q15_1','q15_2','q15_3','q15_4','q15_5','q15_6','q15_7','q15_8','q15_9','q15_10','q15_11','q15_12','q15_13','q15_14','q15_15','q15_16','q15_17','q15_18','q15_19','q15_20',
'q15b_1','q15b_2','q15b_3','q15c','q15d2','q15z','q17_1','q17_2','q17_3','q17_4','q17_5','q17_6','q17_7','q17_8','q17_9','q17_10','q17_11','q17_12','q17_13','q17_14','q17_15','q17_16','q17_17','q17_18','q17_19',
'q24a','q24b','q24c','q26_1','q26_2','q26_3','q26_4','q26_5','q26_6','q26_7','q26_8','q26_9','q26_10','q26_11','q26_12','q26_13','q26_14','q26_15','q26_16','q26_17','q26_18','q26_19','q26_20','q26_21','q26_22','q26_23',
'q27','q28_1','q28_2','q28_3','q28_4','q28_5','q28_6','q28_7','q28_8','q28_9','q28_10','q28_11','q28_12','q28_13',
'q35b_1','q35b_2','q35b_3','q35b_4','q35b_5','q35b_6','q35b_7','q35b_8','q35b_9','q35b_10','q35b_11','q35b_12','q35b_13','q35b_14','q35b_15','q35b_16','q35b_17','q35b_18','q35b_19','q35b_20','q35b_21','q35b_22','q35b_23','q35b_24','q35b_25','q35b_26','q35b_27','q35b_28','q35b_29','q35b_30','q35b_31','q35b_32',
'q53_1','q53_2','q53_3','q53_4','q53_7','q53_8','q53_9','q53_10','q53_11','q53_12','q53_13','q53_14','q53_15','q53_16',
'q36_1_1','q36_1_2','q36_1_3','q36_1_4','q36_1_7','q36_1_8','q36_1_9','q36_1_10','q36_1_11','q36_1_12','q36_1_13','q36_1_14','q36_1_15','q36_1_16','q36_1_18',
'q36_2_1','q36_2_2','q36_2_3','q36_2_4','q36_2_7','q36_2_8','q36_2_9','q36_2_10','q36_2_11','q36_2_12','q36_2_13','q36_2_14','q36_2_15','q36_2_16','q36_2_18',
'q36_3_1','q36_3_2','q36_3_3','q36_3_4','q36_3_7','q36_3_8','q36_3_9','q36_3_10','q36_3_11','q36_3_12','q36_3_13','q36_3_14','q36_3_15','q36_3_16','q36_3_18',
'q36_4_1','q36_4_2','q36_4_3','q36_4_4','q36_4_7','q36_4_8','q36_4_9','q36_4_10','q36_4_11','q36_4_12','q36_4_13','q36_4_14','q36_4_15','q36_4_16','q36_4_18',
'q54_1','q54_2','q54_3','q54_4','q54_7','q54_8','q54_9','q54_10','q54_11','q54_12','q54_13','q54_14','q54_15','q54_16',
'q38_1_1','q38_1_2','q38_1_3','q38_1_4','q38_1_7','q38_1_8','q38_1_9','q38_1_10','q38_1_11','q38_1_12','q38_1_13','q38_1_14','q38_1_15','q38_1_16','q38_1_18',
'q38_2_1','q38_2_2','q38_2_3','q38_2_4','q38_2_7','q38_2_8','q38_2_9','q38_2_10','q38_2_11','q38_2_12','q38_2_13','q38_2_14','q38_2_15','q38_2_16','q38_2_18',
'q38_3_1','q38_3_2','q38_3_3','q38_3_4','q38_3_7','q38_3_8','q38_3_9','q38_3_10','q38_3_11','q38_3_12','q38_3_13','q38_3_14','q38_3_15','q38_3_16','q38_3_18',
'q38_4_1','q38_4_2','q38_4_3','q38_4_4','q38_4_7','q38_4_8','q38_4_9','q38_4_10','q38_4_11','q38_4_12','q38_4_13','q38_4_14','q38_4_15','q38_4_16','q38_4_18',
'q56_1','q56_2','q56_3','q56_4','q56_7','q56_8','q56_9','q56_10','q56_11','q56_12','q56_13','q56_14','q56_15','q56_16',
'q42_1_1','q42_1_2','q42_1_3','q42_1_4','q42_1_7','q42_1_8','q42_1_9','q42_1_10','q42_1_11','q42_1_12','q42_1_13','q42_1_14','q42_1_15','q42_1_16','q42_1_18',
'q42_2_1','q42_2_2','q42_2_3','q42_2_4','q42_2_7','q42_2_8','q42_2_9','q42_2_10','q42_2_11','q42_2_12','q42_2_13','q42_2_14','q42_2_15','q42_2_16','q42_2_18',
'q42_3_1','q42_3_2','q42_3_3','q42_3_4','q42_3_7','q42_3_8','q42_3_9','q42_3_10','q42_3_11','q42_3_12','q42_3_13','q42_3_14','q42_3_15','q42_3_16','q42_3_18',
'q42_4_1','q42_4_2','q42_4_3','q42_4_4','q42_4_7','q42_4_8','q42_4_9','q42_4_10','q42_4_11','q42_4_12','q42_4_13','q42_4_14','q42_4_15','q42_4_16','q42_4_18',
'q60_1','q60_2','q60_3','q60_4','q60_7','q60_8','q60_9','q60_10','q60_11','q60_12','q60_13','q60_14','q60_15','q60_16',
'q43_1_1','q43_1_2','q43_1_3','q43_1_4','q43_1_7','q43_1_8','q43_1_9','q43_1_10','q43_1_11','q43_1_12','q43_1_13','q43_1_14','q43_1_15','q43_1_16','q43_1_18',
'q43_2_1','q43_2_2','q43_2_3','q43_2_4','q43_2_7','q43_2_8','q43_2_9','q43_2_10','q43_2_11','q43_2_12','q43_2_13','q43_2_14','q43_2_15','q43_2_16','q43_2_18',
'q43_3_1','q43_3_2','q43_3_3','q43_3_4','q43_3_7','q43_3_8','q43_3_9','q43_3_10','q43_3_11','q43_3_12','q43_3_13','q43_3_14','q43_3_15','q43_3_16','q43_3_18',
'q43_4_1','q43_4_2','q43_4_3','q43_4_4','q43_4_7','q43_4_8','q43_4_9','q43_4_10','q43_4_11','q43_4_12','q43_4_13','q43_4_14','q43_4_15','q43_4_16','q43_4_18',
'q61_1','q61_2','q61_3','q61_4','q61_7','q61_8','q61_9','q61_10','q61_11','q61_12','q61_13','q61_14','q61_15','q61_16',
'q75_1','q75_2','q75_3','q78','q78b','q78c_1','q78c_2','q78c_3','q78c_4','q78c_5','q78c_6','q81','q81x',
'q84_1','q84_2','q84_3','q84_4','q84_5','q84_6','q84_7','q84_8','q84_9','q84_10','q84_11','q84_12','q84_13','q84_14','q84_15','q84_16','q84_17','q84_18',
'q85','q103106','q111112','q113','qbb2','qbb3','q115','q116_p','q116_l','q117','q119','q120']]


# save the dataframe as a csv file
# df_final.to_csv('./sme_finance_monitor_preprocess.csv')

#####   drop some variables
drop_list=['q36', 'q54']

for i in drop_list:
    col = df_final.filter(like=i).columns.tolist()
    df_final.drop(col,axis=1,inplace=True)

#####   merge variables

df_temp  = df_final.copy()
cols = df_temp.filter(like='q43').columns.tolist()+df_temp.filter(like='q61').columns.tolist()
df_final[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

df_final['q43_61'] = -99.99
for i in cols:
    df_final.loc[df_final[i] == 1.00,'q43_61'] = 1.00
    df_final.loc[df_final[i] == 2.00,'q43_61'] = 2.00
    df_final.drop(i,axis=1,inplace=True)


df_final['q115_116']  = -99.99
df_final.loc[df_final['q115'] == 3.00,'q115_116'] = 0.00
df_final.loc[df_final['q116_p'] == 1.00,'q115_116'] = 1.00
df_final.loc[df_final['q116_p'] == 2.00,'q115_116'] = 2.00
df_final.loc[df_final['q116_p'] == 3.00,'q115_116'] = 3.00
df_final.loc[df_final['q116_p'] == 4.00,'q115_116'] = 4.00
df_final.loc[df_final['q116_p'] == 5.00,'q115_116'] = 5.00
df_final.loc[df_final['q116_l'] == 1.00,'q115_116'] = -1.00
df_final.loc[df_final['q116_l'] == 2.00,'q115_116'] = -2.00
df_final.loc[df_final['q116_l'] == 3.00,'q115_116'] = -3.00
df_final.loc[df_final['q116_l'] == 4.00,'q115_116'] = -4.00
df_final.loc[df_final['q116_l'] == 5.00,'q115_116'] = -5.00

cols = df_final.filter(like='q115').columns.tolist()+df_final.filter(like='q116').columns.tolist()
df_final.drop(cols,axis=1,inplace=True)
#####   variables without missing values
nomiss_list=['risk', 'q126', 'q144', 'q7q8', 'q9', 'q11', 'q12', 'q13', 'q14y', 
'q14ysu2', 'q15d2', 'q15', 'q15z', 'q24a', 'q24b', 'q24c', 'q78', 'q103106', 'q120']

#####   variables with missing values
col_list=['q13b', 'q14a', 'q15_', ['q35b', 'q53'], 'q11a', 'q13a', 'q15b_', 'q15c', 'q17_', 'q26_', 'q27', 
'q28_', ['q38', 'q56'], ['q42', 'q60'], 'q43_61', 'q78b', 'q78c', 'q81', 'q84_', 'q85', 'q111112',
'q113', 'qbb', 'q115_116', 'q117', 'q119']

for i in col_list:
    df_final = multichoice(df_final,i)

df_final.to_csv('./sme_finance_monitor_preprocess.csv')
# #those variables without missing values 
# df_final = pandas.get_dummies(df_final, columns=['risk'])

# df_final = pandas.get_dummies(df_final, columns=['q126'])

# df_final = pandas.get_dummies(df_final, columns=['q144'])

# df_final = pandas.get_dummies(df_final, columns=['q7q8'])

# df_final = pandas.get_dummies(df_final, columns=['q9'])

# df_final = pandas.get_dummies(df_final, columns=['q11'])

# df_final = pandas.get_dummies(df_final, columns=['q12'])

# df_final = pandas.get_dummies(df_final, columns=['q13'])

# df_final = pandas.get_dummies(df_final, columns=['q14y'])

# df_final = pandas.get_dummies(df_final, columns=['q14ysu2'])

# df_final = pandas.get_dummies(df_final, columns=['q15d2'])

# df_final = pandas.get_dummies(df_final, columns=['q15z'])

# df_final = pandas.get_dummies(df_final, columns=['q24a'])

# df_final = pandas.get_dummies(df_final, columns=['q24b'])

# df_final = pandas.get_dummies(df_final, columns=['q24c'])

# df_final = pandas.get_dummies(df_final, columns=['q78'])

# df_final = pandas.get_dummies(df_final, columns=['q103106'])

# df_final = pandas.get_dummies(df_final, columns=['q120'])


# #those variables with missing values but handled 
# df_final = df_final.drop(df_final[df_final['q15_1'] == -99.99].index)
# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q15_').columns.tolist())

# df_temp  = df_final.copy()
# cols = df_temp.filter(like='q35b').columns.tolist()+df_temp.filter(like='q53').columns.tolist()
# df_final[cols] = df_temp[cols].replace(-99.99,0.00,inplace=False)

# #those variables with missing values but not handled 
# df_final = pandas.get_dummies(df_final, columns=['q11a'])

# df_final = pandas.get_dummies(df_final, columns=['q13a'])

# df_final = pandas.get_dummies(df_final, columns=['q13b'])

# df_final = pandas.get_dummies(df_final, columns=['q14a'])

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q15b_').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=['q15c'])

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q17_').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q26_').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=['q27'])

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q28_').columns.tolist())

# df_temp  = df_final.copy()
# cols = df_temp.filter(like='q38').columns.tolist()+df_temp.filter(like='q56').columns.tolist()
# df_final[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

# df_final['q38_56'] = -99.99
# for i in cols:
#     df_final.loc[df_final[i] == 1.00,'q38_56'] = 1.00
#     df_final.loc[df_final[i] == 2.00,'q38_56'] = 2.00
#     df_final.drop(i,axis=1,inplace=True)

# df_final = pandas.get_dummies(df_final, columns=['q38_56'])

# df_temp  = df_final.copy()
# cols = df_temp.filter(like='q42').columns.tolist()+df_temp.filter(like='q60').columns.tolist()
# df_final[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

# df_final['q42_60'] = -99.99
# for i in cols:
#     df_final.loc[df_final[i] == 1.00,'q42_60'] = 1.00
#     df_final.loc[df_final[i] == 2.00,'q42_60'] = 2.00
#     df_final.drop(i,axis=1,inplace=True)

# df_final = pandas.get_dummies(df_final, columns=['q42_60'])

# df_temp  = df_final.copy()
# cols = df_temp.filter(like='q43').columns.tolist()+df_temp.filter(like='q61').columns.tolist()
# df_final[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

# df_final['q43_61'] = -99.99
# for i in cols:
#     df_final.loc[df_final[i] == 1.00,'q43_61'] = 1.00
#     df_final.loc[df_final[i] == 2.00,'q43_61'] = 2.00
#     df_final.drop(i,axis=1,inplace=True)

# df_final = pandas.get_dummies(df_final, columns=['q43_61'])

# df_final = pandas.get_dummies(df_final, columns=['q78b'])

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q78c').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q81').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='q84_').columns.tolist())

# df_final = pandas.get_dummies(df_final, columns=['q85'])

# df_final = pandas.get_dummies(df_final, columns=['q111112'])

# df_final = pandas.get_dummies(df_final, columns=['q113'])

# df_final = pandas.get_dummies(df_final, columns=df_final.filter(like='qbb').columns.tolist())

# df_final['q115_116']  = -99.99
# df_final.loc[df_final['q115'] == 3.00,'q115_116'] = 0.00
# df_final.loc[df_final['q116_p'] == 1.00,'q115_116'] = 1.00
# df_final.loc[df_final['q116_p'] == 2.00,'q115_116'] = 2.00
# df_final.loc[df_final['q116_p'] == 3.00,'q115_116'] = 3.00
# df_final.loc[df_final['q116_p'] == 4.00,'q115_116'] = 4.00
# df_final.loc[df_final['q116_p'] == 5.00,'q115_116'] = 5.00
# df_final.loc[df_final['q116_l'] == 1.00,'q115_116'] = -1.00
# df_final.loc[df_final['q116_l'] == 2.00,'q115_116'] = -2.00
# df_final.loc[df_final['q116_l'] == 3.00,'q115_116'] = -3.00
# df_final.loc[df_final['q116_l'] == 4.00,'q115_116'] = -4.00
# df_final.loc[df_final['q116_l'] == 5.00,'q115_116'] = -5.00

# df_final = pandas.get_dummies(df_final, columns=['q115_116'])

# df_final = pandas.get_dummies(df_final, columns=['q117'])

# df_final = pandas.get_dummies(df_final, columns=['q119'])

# save the dataframe as a csv file
df_final.to_csv('./sme_finance_monitor_final.csv')