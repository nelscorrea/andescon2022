# model_utils.py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import transformers as hf
import tensorflow as tf

import sklearn as sk
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ===========================================================
# Running, saving, loading models

# DEL: get_finetune_model()
def get_finetune_model(model_ckpt, num_labels, train_dataset, val_dataset,  
              batch_size, epochs, model_save_path, model_load_path, finetune_model=False):
    """ Define and fine-tune, or Load fine-tuned model, and save/load model from model_path """
    if finetune_model:
        print(f"Create TF model from `{model_ckpt}` and fine-tune\n save dir: {model_save_path}")
        # Define model
        model = hf.TFAutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, from_pt=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.SparseCategoricalAccuracy())
        # Train model
        print(f"Dataset - batch_size: {batch_size}")
        # print(f"Dataset - train_size: {train_size}, validation_size: {validation_size}, test_size: {test_size}")
        print(f"Model path: {model.name_or_path}, Model name: {model.name}")
        print(f"Fine-tuning model: {model_ckpt}, dir: {model_save_path}") # Wall time: 59h 46min 36s on cpu
        history = model.fit(train_dataset, validation_data=val_dataset, batch_size=batch_size, epochs=epochs)

        print(f"Saving Fine-tuned TF CFPB model: {model_save_path}")
        model.save_pretrained(model_save_path, saved_model=True)
    else:
        print(f"Reading already fine-tuned model: {model_ckpt}\n load dir: {model_load_path}")
        model = hf.TFAutoModelForSequenceClassification.from_pretrained(model_load_path)
        history = None
    return model

# load_model()
def load_model(num_labels, model_ckpt=None, model_load_path=None):
    """ Load original pre-trained model from model_ckpt, or load previously saved model from load_path """
    if model_ckpt:
        print(f"Create TF model from `{model_ckpt}` and fine-tune\n save dir: {model_save_path}")
        # Define model
        model = hf.TFAutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, from_pt=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.SparseCategoricalAccuracy())
    else:
        print(f"Loading model from dir: {model_load_path}")
        model = hf.TFAutoModelForSequenceClassification.from_pretrained(model_load_path)
    return model

# finetune_and_save_model()
def finetune_and_save_model(model, train_dataset, val_dataset, 
                            batch_size, epochs, model_save_path, finetune_model=False):
    """ Fine-tune and save fine-tuned model to model_save_path; return history """
    if finetune_model:
        print(f"Fine-tune TF model `{model.name}` and save to dir: {model_save_path}")
        # Finetune model
        print(f"Dataset - batch_size: {batch_size}")
        print(f"Fine-tuning - model_ckpt: {model.name}, dir: {model_save_path}") # Wall time: 59h 46min 36s on cpu
        history = model.fit(train_dataset, validation_data=val_dataset, batch_size=batch_size, epochs=epochs)
        print(f"Saving Fine-tuned TF CFPB model: {model_save_path}")
        model.save_pretrained(model_save_path, saved_model=True)
    else:
        print(f"Model not fine-tuned\n - model_ckpt: {model.name}, model.name: {model.name}")
        history = None
    return history

# ===========================================================
# Running models and saving and loading model output data

def model_save_output(model_preds_output, saved_data_path):
    """ Save model output to saved_data_path. Returns logits np.array() """
    print(f"Saving logits of model output to saved_data_path: {saved_data_path}")
    preds_logits_df = pd.DataFrame(model_preds_output)
    hd5 = pd.HDFStore(saved_data_path, mode='w') # Save outputs
    hd5["logits"] = preds_logits_df
    logits = np.array(hd5["logits"])
    hd5.close()
    return logits

def model_load_output(saved_data_path):
    """ Load saved model output from saved_data_path. Returns logits np.array() """
    print(f"Loading model output from saved_data_path: {saved_data_path}")
    hd5 = pd.HDFStore(saved_data_path, mode='r') # Save outputs
    logits = np.array(hd5["logits"])
    hd5.close()
    return logits

def model_run_and_save(model, dataset, saved_data_path, ds_name="dataset", run_model=False):
    """ model_run_save(): Run model on `dataset` and save output to saved_data_path; 
    or read from saved_data_path. Returns logits np.array()"""
    if run_model:
        print(f"Running model `{model.name_or_path}` on dataset `{ds_name}`\nfrom saved_data_path: {saved_data_path}")
        model_preds_output = model.predict(dataset)
        logits = model_save_output(model_preds_output.logits, saved_data_path)
    else:
        print(f"Reading saved model output for dataset `{ds_name}`\nfrom saved_data_path: {saved_data_path}")
        logits = model_load_output(saved_data_path)
    return logits

# ===========================================================
# Model Metrics and plotting

# TBD: logloss 
def compute_logloss(y_valid, y_preds):
    logloss = None # from sk.metrics.log_loss(y_valid, y_preds)
    return {"log_loss": logloss}

def compute_metrics(y_valid, y_preds):
    acc = sk.metrics.accuracy_score(y_valid, y_preds)
    p = sk.metrics.precision_score(y_valid, y_preds, average='weighted')
    f1 = sk.metrics.f1_score(y_valid, y_preds, average="weighted")
    return {"accuracy": acc, "precision": p, "f1": f1}

# TBD: Add model_name="" arg
def plot_confusion_matrix(y_preds, y_true, labels, dataset_name="", figsize=(9,6)):
    cm = sk.metrics.confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    disp = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=True, im_kw={"vmin":0.0, "vmax":1.0})
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.title(dataset_name+" - Normalized confusion matrix")
    # plt.imshow(vmin=0.0, vmax=1.0)
    plt.show()
    return

###