import pandas as pd
import numpy as np
import argparse
import re
import os
import sys
import pickle
import glob
import random
import warnings
from pathlib import Path
import torch
import torch.multiprocessing as mp
from transformers import BertTokenizer
warnings.filterwarnings(action='ignore')


def set_seed(seed):
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


def read_table(file_root, filename):
    patient_file_name = os.path.join(file_root, filename)
    patient_data = pd.read_csv(patient_file_name)
    
    if 'Payment_Number' not in patient_data.columns:
        patient_data.rename(columns = {'PaymentID' : 'Payment_Number'}, inplace=True)

    patient_data.Payment_Number = patient_data.Payment_Number.astype(str)
    patient_data.drop_duplicates(inplace=True)
    return patient_data


def extract_common_cohort(opreport, patient_data):
    # Convert to string type for consistent comparison
    opreport.Payment_Number = opreport.Payment_Number.astype(str)
    patient_data.Payment_Number = patient_data.Payment_Number.astype(str)

    # Get the common Payment Numbers
    common_payment_numbers = set(opreport.Payment_Number) & set(patient_data.Payment_Number)

    # Filter both dataframes to include only common Payment Numbers
    patient_data = patient_data[patient_data.Payment_Number.isin(common_payment_numbers)]
    opreport = opreport[opreport.Payment_Number.isin(common_payment_numbers)]

    return opreport, patient_data


class ProcessTable():
    def __init__(self, args, patient_data):
        self.args = args
        self.patient_data = patient_data
        self.file_root = args.output_path
        self.norm = args.norm
        self.categorical_columns = []
        self.numerical_columns = []

        num_null_ids_dict = {'znorm': 0, 'minmax':-1}
        self.num_null_ids = num_null_ids_dict[self.norm]

        
    def remove_useless_column(self):
        self.patient_data.drop('id_for_ordering', axis=1, inplace=True, errors='raise')
        self.patient_data.drop('ID', axis=1, inplace=True, errors='raise')  
        self.patient_data.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore') 
        self.patient_data.drop('사망원인1', axis=1, inplace=True, errors='ignore') 
        self.patient_data.drop('사망원인2', axis=1, inplace=True, errors='ignore') 

        # Drop columns using domain knowledge
        self.patient_data.drop('F_Cr_Max', axis=1, inplace=True, errors='ignore')
        self.patient_data.drop('B_CAPD_ESRD', axis=1, inplace=True, errors='ignore') 

        """As of 23.06.16, we changed the columns used."""
    
        # 1. Columns that can be created with logic are not synthesized.
        columns_to_drop = ['A_BMI', 'B_eGFR_EPI_near', 'B_eGFR_EPI_stage_near']
        self.patient_data.drop(columns_to_drop, axis=1, inplace=True, errors='raise')

        # 2. Columns that is a higher group in the definition process
        columns_to_drop = ['A_SMK1', 'AN_DM', 'AN_HTN', 'B_DM', 'B_HTN', 'O_AKI_criteria1', 'O_AKI_criteria2', 'O_HD_7','O_HD_90', 'O_CRRT_7','O_CRRT_90']
        self.patient_data.drop(columns_to_drop, axis=1, inplace=True, errors='raise')

        columns_to_drop = ['AKI_Cr2_Days_after_Op_start', 'AKI_first_Stage', 'AKI_Max_Stage', 'AKI_Cr2_Cr_diff', 'AKI_Cr2_Value', 'AKI_Max_Value', 'AKI_first_Value']
        self.patient_data.drop(columns_to_drop, axis=1, inplace=True, errors='raise')


    def process_obj(self, obj_patient_data):
        # 1. Remove the date,time-related column ** EXCEPT Op_Date **
        opdate = pd.to_datetime(obj_patient_data['Op_Date']).dt.year

        total_columns = set(obj_patient_data.columns)
        date_columns = set(obj_patient_data.filter(like='date').columns) \
                        | set(obj_patient_data.filter(like='Date').columns) \
                        | set(obj_patient_data.filter(like='time').columns) \
                        | set(obj_patient_data.filter(like='Time').columns)

        non_date_columns = total_columns - date_columns
        obj_patient_data = obj_patient_data[non_date_columns]

        # ** EXCEPT Op_Date **
        obj_patient_data['Op_Year'] = opdate

        # 2. Filter for Hepatectomy dataset (old version)
        first_data_useless_columns = ['A_Birth', 'First_CRRT', 'Op_EST_Dur', 'Last_CRRT', 'ICU_ADM', 'ICU_detailed', 'ICU_DISCHARGE']
        for column_name in first_data_useless_columns:
            if column_name in non_date_columns:
                obj_patient_data.drop(column_name, axis=1, inplace=True)

        # 3. Change both Op_Name and Op_Dx to lowercase letters
        if 'Op_Name' in obj_patient_data.columns:
            obj_patient_data['Op_Name'] = obj_patient_data['Op_Name'].str.lower()
        if 'Op_Dx' in obj_patient_data.columns:
            obj_patient_data['Op_Dx'] = obj_patient_data['Op_Dx'].str.lower()

        # 4. Change Op_Dur to rounded hour
        if 'Op_Dur' in obj_patient_data.columns:
            obj_patient_data['Op_Dur'] = pd.to_datetime(obj_patient_data['Op_Dur'])
            obj_patient_data['Op_Dur'] = obj_patient_data['Op_Dur'].round('H').dt.hour

        # 5. # of (Op_Name.str.lower) = 13682 -> Too much
        obj_patient_data.drop('Op_Name', axis= 1, inplace= True)

        # 6. Split columns into categorical/numeric (Set all columns to categorical except Op_Dur)
        col_list = list(obj_patient_data.columns)
        col_list.remove('Payment_Number')
        if 'Op_Dur' in col_list:
            col_list.remove('Op_Dur')
        self.categorical_columns.append(col_list)
        print(f'# obj categorical columns : {len(col_list)}')
        return obj_patient_data


    def process_int(self, int_patient_data):
        # 1. Remove columns (nunique==1)
        mask = (int_patient_data.nunique() == 1).values
        int_patient_data = int_patient_data[int_patient_data.columns[~mask]]

        # 2. Leave the column of nunique==2 as a categorical column
        mask = (int_patient_data.nunique() == 2).values

        # 3-1. Classify columns (nunique > 10) into numeric
        # 3-2. Classify columns (2 < nunique < 10) into categorical
        temp = int_patient_data[int_patient_data.columns[~mask]]
    
        col_list = list(set(int_patient_data.columns) - set(temp.columns[(temp.nunique() >= 10).values].values))
        self.categorical_columns.append(col_list)
        print(f'# int categorical columns : {len(col_list)}')
        return int_patient_data


    def process_float(self, float_patient_data):
        # 1. Remove 100% null columns
        null_columns = [col for col in float_patient_data.columns if float_patient_data[col].isnull().sum() == float_patient_data.shape[0]]
        non_null_columns = set(float_patient_data.columns) - set(null_columns)
        float_patient_data = float_patient_data[non_null_columns]
        
        # 2. Leave the column of nunique < 10 as a categorical column
        mask = (float_patient_data.nunique() < 10).values
        float_patient_data[float_patient_data.columns[mask]].nunique()

        col_list = list(float_patient_data.columns[mask])
        self.categorical_columns.append(col_list)
        print(f'# float categorical columns : {len(col_list)}')
        return float_patient_data


    def merge_columns(self, obj_patient_data, int_patient_data, float_patient_data):
        df = pd.concat([obj_patient_data, int_patient_data, float_patient_data], axis=1)

        # Define categorical columns and numerical columns
        categorical_columns = sum(self.categorical_columns, [])
        numerical_columns = list(set(df.columns) - set(categorical_columns))
        numerical_columns.remove('Payment_Number')

        categorical_columns = sorted(categorical_columns)
        numerical_columns = sorted(numerical_columns)
        print(f'categorical : numeric = {len(categorical_columns)} : {len(numerical_columns)}')
        
        # Map categorical columns to have a cumulative labels
        vocab = 0
        cat_columns_numclass_list = []
        class2raw = {}
        for column in categorical_columns:
            column_unique_vocab = df[column].dropna().unique()        
            raw2class = {string : int(i+vocab) for i,string in enumerate(column_unique_vocab)}
            for k, v in raw2class.items():
                class2raw[v] = k
            df[column] = df[column].map(raw2class)
            vocab += len(column_unique_vocab)
            cat_columns_numclass_list.append(len(column_unique_vocab))

        with open(os.path.join(args.output_path, args.norm, 'class2raw.pickle'),'wb') as fw:
            pickle.dump(class2raw, fw)
        
        # Fill null with max vocab in categorical columns
        df[categorical_columns] = df[categorical_columns].fillna(value=vocab)
        df[categorical_columns] = df[categorical_columns].astype(int)
        print(f'Categorical NaN id = {vocab}' )
        
        # Normalize numerical columns 
        unnorm_df = df.copy()
        file_path = os.path.join(self.file_root, self.norm, '{}.csv'.format(self.norm)) 

        if self.norm == 'minmax':
            normalization_info = pd.DataFrame({
                'min': df[numerical_columns].min(),
                'max': df[numerical_columns].max()
            })
            normalization_info.to_csv(file_path)
            df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].min()) / (
                    df[numerical_columns].max() - df[numerical_columns].min())
        elif self.norm == 'znorm':
            normalization_info = pd.DataFrame({
                'mean': df[numerical_columns].mean(),
                'std': df[numerical_columns].std()
            })
            normalization_info.to_csv(file_path)
            df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()
        else:
            raise ValueError("Invalid normalization method specified.")

        # Fill null with -1 in numeric columns
        df[numerical_columns] = df[numerical_columns].fillna(value=self.num_null_ids)
        unnorm_df[numerical_columns] = unnorm_df[numerical_columns].fillna(value=self.num_null_ids)
        
        # Arrange table in order of Payment_Number
        df.sort_values('Payment_Number', inplace=True)
        unnorm_df.sort_values('Payment_Number', inplace=True)

        # For convenience, fix the column order.
        column_order = ['Payment_Number'] + categorical_columns + numerical_columns
        df = df[column_order]
        unnorm_df = unnorm_df[column_order]

        info_dict = {
            'cat_vocab_size' : vocab+1,
            'cat_col_num' : len(categorical_columns),
            'num_col_num' : len(numerical_columns),
            'cat_null_ids': vocab,
            'num_null_ids' : self.num_null_ids,
            'column_order': column_order[1:],
            'cat_columns_numclass_list':cat_columns_numclass_list
        }
        return df, unnorm_df, info_dict


    def overall_process(self):
        self.remove_useless_column()

        # Process
        raw_obj_patient_data = self.patient_data.select_dtypes(include=['object'])
        raw_int_patient_data = self.patient_data.select_dtypes(include=['int64'])
        raw_float_patient_data = self.patient_data.select_dtypes(include=['float64'])
        
        obj_patient_data = self.process_obj(raw_obj_patient_data)
        int_patient_data = self.process_int(raw_int_patient_data)
        float_patient_data = self.process_float(raw_float_patient_data)

        print(f'obj columns : {raw_obj_patient_data.shape} -> {obj_patient_data.shape}')
        print(f'int columns : {raw_int_patient_data.shape} -> {int_patient_data.shape}')
        print(f'float columns : {raw_float_patient_data.shape} -> {float_patient_data.shape}')

        # Merge
        table_df, unnorm_table_df,  info_dict = self.merge_columns(
                                                        obj_patient_data,
                                                        int_patient_data,
                                                        float_patient_data,
                                                    )
        return table_df, unnorm_table_df, info_dict


def read_op(file_root, pat):
    # Load data from SPARK files
    spark_files = [os.path.join(file_root, f'spark/SPARK_ID{i}.xlsx') for i in range(1, 6)]
    spark_data = pd.concat([pd.read_excel(file, skiprows=[0]) for file in spark_files], axis=0)
    print(f'SPARK data loaded! Shape: {spark_data.shape}')
    
    # Load data from synthesis files
    synthesis_files = [os.path.join(file_root, f'synthesis/ID{i}.xlsx') for i in range(1, 17)]
    synthesis_data = pd.concat([pd.read_excel(file, skiprows=[0]) for file in synthesis_files], axis=0)
    print(f'Synthesis data loaded! Shape: {synthesis_data.shape}')

    # Combine SPARK and synthesis data
    op = pd.concat([spark_data, synthesis_data], axis=0)
    print(f'op: {op.shape}')

    # Keep only common columns
    common_columns = set(spark_data.columns) & set(synthesis_data.columns)
    op = op[common_columns]

    op.rename(columns={'원무접수ID': 'Payment_Number', '환자번호': 'ID'}, inplace=True)
    op.Payment_Number = op.Payment_Number.astype(str)
    op.ID = op.ID.astype(str)
    op.drop_duplicates(inplace=True)
    return op
    

class process_text:
    def __init__(self, op, len_input=1000, num_table=134):
        self.op = op
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer.max_model_input_sizes[ 'bert-base-multilingual-cased'] = None
        self.num_vocab = self.tokenizer.vocab_size
        self.len_input = len_input 
        self.num_table = num_table
        self.num_txt = self.len_input - self.num_table
    
    def basic_pp(self):
        df = self.op[['Payment_Number', '서식항목명', '서식내용']].copy()
        df['서식내용'] = df['서식내용'].apply(str)
        df['서식내용'] = df['서식내용'].apply(lambda x: re.sub(r"[-=+, \?^$@\"※ㆍ!』\r\n\\‘|\[\]`…》]", " ", x))
        df['서식항목명'] = df['서식항목명'].apply(lambda x: x.replace('<-', '@'))        
        return df
    
    def filter_procedure(self, df):
        df2 = df.copy()
        df_half1 = df2[df2['서식항목명'] == 'Procedure @ 수술방법 및 소견']
        df_half2 = df2[df2['서식항목명'] != 'Procedure @ 수술방법 및 소견']
        print(f'the length of 서식내용 before procedure: {df_half1.서식내용.map(len).mean()}')

        # Delete the common procedure parts in 서식내용
        df_half1['서식내용'] = df_half1['서식내용'].apply(lambda x: x.split('      '))
        df_half1['서식내용'] = df_half1['서식내용'].apply(lambda x: [i for i in x if not i.startswith('#')])
        df_half1['서식내용'] = df_half1['서식내용'].apply(lambda x: ' '.join(x).strip())
        df_half1 = df_half1[df_half1['서식내용'] != '']
        print(f'the length of 서식내용 before procedure: {df_half1.서식내용.map(len).mean()}')

        df2 = pd.concat([df_half1, df_half2])        
        return df2
    
    def select_op_findings(self, op):
        op_findings = op[op['서식항목명'].apply(lambda x: '수술방법 및 소견' in x)].copy()
        return op_findings
    
    def name_compression(self, op_findings):
        # removing the string <- 수술방법 및 소견'
        op_findings['서식항목명'] = op_findings['서식항목명'].str.replace('@ 수술방법 및 소견', '')
        return op_findings
    
    def merge_tokenize(self, df_input):
        df = df_input.copy()
        tokenizer = self.tokenizer

        # a sentence from each row = 서식항목명 $ 서식내용
        df['sentence'] = df['서식항목명'].apply(str) + ' $ ' + df['서식내용']

        # grouping the sentences from same patient.
        df_sentence = df.groupby('Payment_Number')['sentence'].apply(list).apply(lambda x: ' [SEP] '.join(x))

        # filtering procedure
        def filter_procedure(row):
            l = row.split('    ')
            return ' '.join(list(filter(lambda x: not x.strip().startswith('#'), l)))
        
        df_sentence = df_sentence.apply(filter_procedure)
        df_sentence = df_sentence.reset_index()
        
        df_tokenized = pd.DataFrame()
        df_tokenized['Payment_Number'] = df_sentence['Payment_Number']        
        df_token = df_sentence['sentence'].apply(
            lambda x: tokenizer.encode(x, max_length= self.num_txt, truncation=True, padding='max_length')).values
        print('    tokenize - done!')
                
        df_tokenized['sentence'] = df_sentence['sentence']
        df_tokenized['token_ids_trun_pad'] = df_token
        df_tokenized = df_tokenized[['Payment_Number', 'token_ids_trun_pad']]
        return df_tokenized

    
    def overall_process(self):
        df = self.basic_pp()
        print('op: basic_pp - done!')
        df = self.select_op_findings(df)
        print('op: select_op_findings - done!')
        df = self.name_compression(df)
        print('op: name_compression - done!')
        df = self.merge_tokenize(df)
        print('op: merge_tokenize - done!')
        return df


def merge_text_table(table_df, unnorm_table_df, file_root, sub_root, cat_null_ids, num_null_ids, cat_col_num, num_col_num, text_df=None):
    def save_file(file_name, file_content):
        file_path = os.path.join(file_root, sub_root, file_name)
        if isinstance(file_content, pd.DataFrame):
            file_content.to_csv(file_path, index=False)
        else:
            np.save(file_path, file_content)
        print(f'{file_name} : ', file_content.shape)

    if text_df is not None:
        text_df['token_ids_trun_pad'] = text_df['token_ids_trun_pad'].apply(eval)
        text_df = pd.concat([text_df["token_ids_trun_pad"].apply(pd.Series), text_df['Payment_Number']], axis=1)

        # Merge text_df with table_df
        df = pd.merge(table_df, text_df, on="Payment_Number")
        unnorm_table_df = pd.merge(unnorm_table_df, text_df, on="Payment_Number")
        text_tokens = np.ones((df.shape[0], text_df.shape[1]-1), dtype=np.int32)
    else:
        df = table_df.copy()
        text_tokens = np.array([])
    
    df.sort_values(by=['Payment_Number'], inplace=True)
    df.drop('Payment_Number', axis=1, inplace=True)
    
    # Save original DataFrame
    save_file('original_df.csv', df)

    # input_ids
    input_ids = df.to_numpy()
    save_file('input_ids.npy', input_ids)

    # unnormalize input_ids
    unnorm_table_df.sort_values(by=['Payment_Number'], inplace=True)
    unnorm_table_df.drop('Payment_Number', axis= 1, inplace= True)

    unnormalized_input_ids = unnorm_table_df.to_numpy()
    save_file('unnormalized_input_ids.npy', unnormalized_input_ids)
    
    # Calculate null_type_ids
    null_type_ids = np.concatenate([(input_ids[:, :cat_col_num] == cat_null_ids),
                                    (input_ids[:, cat_col_num:cat_col_num + num_col_num] == num_null_ids)],
                                   axis=1)
    save_file('null_type_ids.npy', null_type_ids)
    
    # token_type_ids
    table_tokens = np.zeros((df.shape[0], table_df.shape[1]-1), dtype=np.int32)
    token_type_ids = np.concatenate([table_tokens, text_tokens], axis=1) if text_df is not None else table_tokens
    save_file('token_type_ids.npy', token_type_ids)

    return table_df.shape[1]-1, text_df.shape[1]-1 if text_df is not None else None


def fold_split(total_data, target_fold_root):
    seed_list = [2020, 2021, 2022, 2023, 2024]

    ind_list = np.arange(total_data.shape[0])
    total_num = len(ind_list)

    train_valid_num = int(total_num * 0.9)
    test_num = total_num - train_valid_num
    train_num = int(train_valid_num * 0.8)
    valid_num = train_valid_num - train_num
    print('total num:', total_num, 'train num:', train_num, 'valid num:',valid_num, 'test num:',test_num, 'total num:',train_num +valid_num + test_num)
    
    for seed in seed_list:
        
        data = np.zeros(total_num).astype(np.int32)
        ind_list = np.arange(total_data.shape[0])
        
        np.random.seed(seed)
        np.random.shuffle(ind_list)
        
        data[ind_list[:train_num]] = 1
        data[ind_list[train_num:train_num+valid_num]] = 2
        data[ind_list[train_num+valid_num:]] = 0

        target_fold_path = os.path.join(target_fold_root, 'snuh_{}_fold_split.csv'.format(seed))
        df = pd.DataFrame({'fold': data})
        df.to_csv(target_fold_path, index=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--preprocessed_op_filename', type=str, required=True)
    parser.add_argument('--patient_data_filename', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--norm', type=str, default='znorm', choices=['znorm', 'minmax'])
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--use_text', action='store_true') 
    return parser


if __name__ == '__main__':ㄴ
    args = get_parser().parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, args.norm), exist_ok=True)

    patient_data = read_table(args.input_path, args.patient_data_filename)
    text_df = None

    if args.use_text:
        if not os.path.isfile(os.path.join(args.input_path, args.preprocessed_op_filename)):
            op = read_op(args.input_path, patient_data)
            op_report_class = process_text(op)
            op_df = op_report_class.overall_process()
            op_df.to_csv(os.path.join(args.input_path, args.preprocessed_op_filename), index=False)   
            print('>>> Text processing is done ..')
        else:            
            text_df = pd.read_csv(os.path.join(args.input_path, args.preprocessed_op_filename))
            text_df, patient_data = extract_common_cohort(text_df, patient_data)

    table_class = ProcessTable(args, patient_data) 
    table_df, unnorm_table_df, info_dict = table_class.overall_process()
    print('>>> Table processing is done ..')

    ntable, ntext = merge_text_table(
        table_df=table_df,
        unnorm_table_df=unnorm_table_df,
        file_root=args.output_path,
        sub_root=args.norm,
        cat_null_ids=info_dict['cat_null_ids'],
        num_null_ids=info_dict['num_null_ids'],
        cat_col_num=info_dict['cat_col_num'],
        num_col_num=info_dict['num_col_num'],
        text_df=text_df
    )
    print('>>> Merging text and table is done ..')

    # Split folds
    data = np.load(os.path.join(args.output_path, args.norm, 'input_ids.npy'))
    fold_root= os.path.join(args.output_path, args.norm, 'fold')
    os.makedirs(fold_root, exist_ok=True)
    fold_split(data, fold_root)

    print('>>> Fold splits are done ..')
    print(os.listdir(fold_root))

    # Save argument
    assert ntable == (info_dict['cat_col_num']+info_dict['num_col_num'])

    info_dict['text_num'] = ntext
    pickle_path = os.path.join(args.output_path, args.norm, 'info_dict.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(info_dict, f)

    print('>>> Saving arguments is done ..')
