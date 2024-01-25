from sacred import Experiment

ex = Experiment("METER", save_git_info=False)

@ex.config
def config():
    
    ###################### Make your own configs ######################
    input_path = ''
    output_path = ''
    
    norm = 'znorm'
    test_sets = ['test']

    nc = 68 # number of categorical columns
    nn = 40 # number of numerical columns
    ntext = 866 # number of text tokens
    num_null_id = 0
    cat_null_id = 2480
    cat_vocab_size = 2481+2 #2 for pad, mask
    column_order = ['AB_DM', 'AB_HTN', 'AN_ASA', 'AN_Asthma', 'AN_COPD', 'AN_Heart_Dz', 'AN_Hematologic_Dz', 'AN_Liver_Dz', 'AN_NYHA', 'AN_Neurologic_Dz', 'AN_Other_organ_Dz', 'AN_Pregnancy', 'AN_Renal_Dz', 'AN_TB', 'AN_Thyroid_Dz', 'AN_Vascular_Dz', 'A_SMK2', 'A_Sex', 'B_AKI', 'B_CAD', 'B_CKD', 'B_COPD', 'B_CVD', 'B_Malig', 'B_UALB', 'B_URBC', 'CA_AKI_2W', 'D_Aspirin_14', 'D_Aspirin_90', 'D_Clopidogrel_14', 'D_Clopidogrel_90', 'D_DIURETICS14', 'D_DIURETICS90', 'D_Ezetimibe_14', 'D_Ezetimibe_90', 'D_Fenofibrate_14', 'D_Fenofibrate_90', 'D_ISA_14', 'D_ISA_90', 'D_LMWH_14', 'D_LMWH_90', 'D_NOAC_14', 'D_NOAC_90', 'D_NSAID14', 'D_NSAID90', 'D_RASB_14', 'D_RASB_90', 'D_Statin_14', 'D_Statin_90', 'D_Steroid_14', 'D_Steroid_90', 'D_Warfarin_14', 'D_Warfarin_90', 'O_AKI', 'O_AKI_stage', 'O_Critical_AKI_7', 'O_Critical_AKI_90', 'O_Death_7', 'O_Death_90', 'O_RRT_7', 'O_RRT_90', 'Op_AN', 'Op_Code', 'Op_Dep', 'Op_Type', 'Op_Year', 'Study', 'Type_Adm', 'A_Age', 'A_DBP', 'A_HR', 'A_HT', 'A_SBP', 'A_WT', 'B_ALP', 'B_ALT', 'B_AST', 'B_Alb', 'B_BIL', 'B_BUN', 'B_CL', 'B_Ca', 'B_Chol', 'B_Cr_near', 'B_ESR', 'B_Glucose', 'B_HDL', 'B_Hb', 'B_HbA1c', 'B_Hct', 'B_INR', 'B_K', 'B_LDL', 'B_Na', 'B_Neutrophil', 'B_P', 'B_PTH', 'B_Plt', 'B_Protein', 'B_Triglyceride', 'B_UPCR', 'B_Uric', 'B_WBC', 'B_hsCRP', 'B_tCO2', 'Dur_Adm', 'Op_Dur', 'Op_EST_Dur']
    cat_columns_numclass_list = [2, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2320, 5, 2, 13, 2, 2]
    
    #######################################################################

    tokenizer = 'bert-base-multilingual-cased'
    text_vocab_size = 119547+1 #bert-base-multilingual-cased + eos token
    soe_token = 101
    sep_token = 102
    eos_token = 119547

    #Model settings
    modeling = 'AR'
    sep_regressor = False
    sep_numemb = False
    bert_type = "mini"
    var_head_type = 'abs'

    #Training settings
    n_epochs = 500
    patience = 30
    dropout = 0.1

    #Below params varies with the environment
    modality = ['table']
    lr = 5e-5
    seed = 2020

    num_nodes = 1 
    num_gpus = 1
    per_gpu_batchsize = 16

    test_only = False
    resume = False
    debug = False
    
    save_dir = 'checkpoints'
    wandb_project_name = "SNUH text generation"
    temperature = 1.0
    topk = None
    topp = None
    topk_filter_thres = 0.9
    sample=False
    prevent_repeat_ngram=0
    prevent_too_short=0
    num_samples = 1000

@ex.named_config
def task_train_both2text():
    modality = ['table', 'text']

@ex.named_config
def task_generate_both2text():
    modality = ['table', 'text']
    per_gpu_batchsize = 50
    
    debug = True
    sample = True
    test_only = True

@ex.named_config
def task_train_text2text():
    modality = ['text']
