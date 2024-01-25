# MedTable2OpText
A repository for generating postoperative records by referencing pre-surgery patient table information.
## How to use

### Step 1: Data Preprocessing
Firstly, prepare your data:
1. Remove unnecessary columns.
2. Categorize columns into categorical or numerical types.
3. Normalize numerical columns.

Use the provided script to automate this process. If your data includes text, use the `--use_text` flag:

```python
python mmtg/datamodules/preprocess.py \
--input_path={input_path} \
--preprocessed_op_filename={preprocessed_op_filename} \
--patient_data_filename={patient_data_filename} \
--output_path={output_path} \
--use_text
```

### Step 2: Configuration
Fill in the mmtg/config.py file according to the output of step 1.

### Step 3: Model Training
Train your model using the following command:
```bash
bash run_scripts/train.sh
```

### Step 4: Data Generation
Generate postoperative records using your trained model:  
```bash
bash run_scripts/generate.sh
```