#!/usr/bin/env python3
"""
FedMed-BN: Data Preprocessing for Bangla Medical NER
Converts 3 datasets into unified BIO format for federated learning
"""

import pandas as pd
import json
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

# ============================================================
# DATASET 1: Bengali Medical Named Entity Recognition.csv
# Format: Patient ID, token, Tag, Gazetteers
# Tags: Bp, S, A, V, T, C, D, F, Bl, E, O, N, Y
# ============================================================
def load_dataset1(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Dataset 1 shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique tags: {df['Tag'].unique()}")
    print(f"Tag distribution:\n{df['Tag'].value_counts()}")
    return df

def convert_dataset1_to_bio(df):
    """Convert patient-grouped tokens to sentence-level BIO format"""
    # Group by Patient ID first, then split by punctuation
    sentences = []
    
    for patient_id, group in df.groupby("Patient ID"):
        current_sentence = []
        
        for _, row in group.iterrows():
            token = str(row["Problem's token"]).strip()
            tag = str(row["Tag"]).strip()
            
            if not token or token == "nan":
                continue
            
            # Sentence boundary detection for Bangla
            if token in ["।", "?", "!", "।", ".", "?"]:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            # Map tags to medical entity types
            tag_map = {
                "Bp": "ORGAN",      # Body part
                "S": "SYMPTOM",     # Symptom
                "F": "FLUID",       # Fluid
                "Bl": "FLUID",      # Blood
                "E": "EXCRETION",   # Excretion
                "C": "COLOR",       # Color
                "D": "DIRECTION",   # Direction
            }
            
            if tag in tag_map:
                entity = tag_map[tag]
                if current_sentence and current_sentence[-1][1].startswith("I-"):
                    bio_tag = f"I-{entity}"
                else:
                    bio_tag = f"B-{entity}"
            elif tag in ["A", "V", "T", "O", "nan", "Y", "N"]:
                bio_tag = "O"
            else:
                bio_tag = "O"
                
            current_sentence.append((token, bio_tag))
        
        if current_sentence:
            sentences.append(current_sentence)
            current_sentence = []
    
    print(f"Dataset 1: {len(sentences)} sentences extracted")
    return sentences


# ============================================================
# DATASET 2: medical_entity_dataset v2.xlsx
# Format: Text, Medical Entity, Label
# Labels: Medicine, Disease, Common Medical Terms, Pharmacological Class, Organ, Hormone
# ============================================================
def load_dataset2(xlsx_path):
    df = pd.read_excel(xlsx_path)
    print(f"Dataset 2 shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    return df

def convert_dataset2_to_bio(df):
    """Convert entity-level annotations to sentence-level BIO"""
    # Group by Text to reconstruct sentences
    sentences = []
    
    for text, group in df.groupby("Text"):
        text = str(text).strip()
        if not text:
            continue
            
        # Simple tokenization for Bangla (space + punctuation)
        tokens = re.findall(r'\S+', text)
        if not tokens:
            continue
            
        labels = ["O"] * len(tokens)
        
        # Find entities in text and mark BIO
        for _, row in group.iterrows():
            entity = str(row["Medical Entity"]).strip()
            label = str(row["Label"]).strip()
            
            if entity in text:
                # Find position of entity in tokens
                entity_tokens = re.findall(r'\S+', entity)
                if not entity_tokens:
                    continue
                    
                # Simple matching - find first occurrence
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        for j, et in enumerate(entity_tokens):
                            if j == 0:
                                labels[i+j] = f"B-{label}"
                            else:
                                labels[i+j] = f"I-{label}"
                        break
        
        sentence = list(zip(tokens, labels))
        sentences.append(sentence)
    
    print(f"Dataset 2: {len(sentences)} sentences extracted")
    return sentences


# ============================================================
# DATASET 3: annotated_entity_11000_V0.txt
# Format: token<TAB>tag (BIO-style: AN_B, AN_I, DS_B, DS_I, O)
# ============================================================
def load_dataset3(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Dataset 3: {len(lines)} lines")
    return lines

def convert_dataset3_to_bio(lines):
    """Convert tab-separated token-tag to sentence-level BIO"""
    sentences = []
    current = []
    
    tag_map = {
        "AN_B": "B-ANATOMY",
        "AN_I": "I-ANATOMY",
        "DS_B": "B-DISEASE",
        "DS_I": "I-DISEASE",
        "O": "O"
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
            
        parts = line.split("\t")
        if len(parts) >= 2:
            token = parts[0].strip()
            tag = parts[1].strip()
            bio_tag = tag_map.get(tag, "O")
            current.append((token, bio_tag))
            
            # Sentence boundary: Bangla full stop
            if token in ["।", "?", "!", "."]:
                if current:
                    sentences.append(current)
                    current = []
    
    if current:
        sentences.append(current)
    
    print(f"Dataset 3: {len(sentences)} sentences extracted")
    return sentences


# ============================================================
# UNIFY ALL DATASETS
# ============================================================
def unify_tags(sentences, dataset_name):
    """Map all tags to unified entity types"""
    # Target entity types for FedMed-BN:
    # DISEASE, MEDICINE, ORGAN, HORMONE, PHARMACOLOGICAL_CLASS, COMMON_MEDICAL_TERMS, ANATOMY, SYMPTOM
    tag_mapping = {
        # Dataset 1
        "B-ORGAN": "B-ORGAN", "I-ORGAN": "I-ORGAN",
        "B-SYMPTOM": "B-SYMPTOM", "I-SYMPTOM": "I-SYMPTOM",
        "B-FLUID": "B-COMMON_MEDICAL_TERMS", "I-FLUID": "I-COMMON_MEDICAL_TERMS",
        "B-EXCRETION": "B-COMMON_MEDICAL_TERMS", "I-EXCRETION": "I-COMMON_MEDICAL_TERMS",
        "B-COLOR": "B-COMMON_MEDICAL_TERMS", "I-COLOR": "I-COMMON_MEDICAL_TERMS",
        "B-DIRECTION": "B-COMMON_MEDICAL_TERMS", "I-DIRECTION": "I-COMMON_MEDICAL_TERMS",
        
        # Dataset 2
        "B-Disease": "B-DISEASE", "I-Disease": "I-DISEASE",
        "B-Medicine": "B-MEDICINE", "I-Medicine": "I-MEDICINE",
        "B-Common Medical Terms": "B-COMMON_MEDICAL_TERMS", "I-Common Medical Terms": "I-COMMON_MEDICAL_TERMS",
        "B-Pharmacological Class": "B-PHARMACOLOGICAL_CLASS", "I-Pharmacological Class": "I-PHARMACOLOGICAL_CLASS",
        "B-Organ": "B-ORGAN", "I-Organ": "I-ORGAN",
        "B-Hormone": "B-HORMONE", "I-Hormone": "I-HORMONE",
        
        # Dataset 3
        "B-ANATOMY": "B-ORGAN", "I-ANATOMY": "I-ORGAN",
        "B-DISEASE": "B-DISEASE", "I-DISEASE": "I-DISEASE",
        "O": "O"
    }
    
    unified = []
    for sent in sentences:
        new_sent = []
        for token, tag in sent:
            new_tag = tag_mapping.get(tag, "O")
            new_sent.append((token, new_tag))
        unified.append(new_sent)
    return unified


def save_bio_format(sentences, output_path):
    """Save in CoNLL format (token tag per line, blank line between sentences)"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for token, tag in sent:
                f.write(f"{token} {tag}\n")
            f.write("\n")
    print(f"Saved {len(sentences)} sentences to {output_path}")


def create_non_iid_partitions(sentences, num_clients=3, seed=42):
    """
    Create non-IID partitions for 3 simulated hospitals:
    - Hospital A: Disease/Symptom heavy (diagnostic center)
    - Hospital B: Medicine/Pharmacological heavy (pharmacy/prescription)
    - Hospital C: Organ/Anatomy heavy (surgery/specialist)
    """
    np.random.seed(seed)
    
    # Categorize sentences by dominant entity type
    disease_sents = []
    medicine_sents = []
    organ_sents = []
    other_sents = []
    
    for sent in sentences:
        tags = [tag for _, tag in sent]
        entity_tags = [t for t in tags if t != "O"]
        
        if not entity_tags:
            other_sents.append(sent)
            continue
            
        # Count entity types
        disease_count = sum(1 for t in entity_tags if "DISEASE" in t or "SYMPTOM" in t)
        medicine_count = sum(1 for t in entity_tags if "MEDICINE" in t or "PHARMACOLOGICAL" in t)
        organ_count = sum(1 for t in entity_tags if "ORGAN" in t or "ANATOMY" in t)
        
        if disease_count >= medicine_count and disease_count >= organ_count:
            disease_sents.append(sent)
        elif medicine_count >= organ_count:
            medicine_sents.append(sent)
        else:
            organ_sents.append(sent)
    
    print(f"Disease-heavy: {len(disease_sents)}")
    print(f"Medicine-heavy: {len(medicine_sents)}")
    print(f"Organ-heavy: {len(organ_sents)}")
    print(f"Other: {len(other_sents)}")
    
    # Distribute with skew
    client_data = [[] for _ in range(num_clients)]
    
    # Hospital 0: 60% disease, 20% medicine, 10% organ, 10% other
    # Hospital 1: 10% disease, 60% medicine, 20% organ, 10% other
    # Hospital 2: 20% disease, 10% medicine, 60% organ, 10% other
    
    distributions = [
        (0.6, 0.2, 0.1, 0.1),  # Hospital A
        (0.1, 0.6, 0.2, 0.1),  # Hospital B
        (0.2, 0.1, 0.6, 0.1),  # Hospital C
    ]
    
    pools = [disease_sents, medicine_sents, organ_sents, other_sents]
    
    for client_idx, (d_w, m_w, o_w, ot_w) in enumerate(distributions):
        for pool_idx, (pool, weight) in enumerate(zip(pools, [d_w, m_w, o_w, ot_w])):
            n = int(len(pool) * weight / num_clients) + 1
            np.random.shuffle(pool)
            client_data[client_idx].extend(pool[:n])
    
    # Shuffle each client's data
    for client_idx in range(num_clients):
        np.random.shuffle(client_data[client_idx])
        print(f"Client {client_idx}: {len(client_data[client_idx])} sentences")
    
    return client_data


def split_train_test(client_data, test_ratio=0.2):
    """Split each client's data into train/test"""
    train_data = []
    test_data = []
    for client_sentences in client_data:
        train, test = train_test_split(client_sentences, test_size=test_ratio, random_state=42)
        train_data.append(train)
        test_data.append(test)
    return train_data, test_data


def main():
    # Paths - adjust as needed
    base = r"C:\Users\muhib\Desktop\New folder - Copy\downloaded datasets"
    
    # Load all datasets
    df1 = load_dataset1(f"{base}\\Bengali Medical Named Entity Recognition.csv")
    df2 = load_dataset2(f"{base}\\medical_entity_dataset v2.xlsx")
    lines3 = load_dataset3(f"{base}\\annotated_entity_11000_V0.txt")
    
    # Convert to BIO
    sents1 = convert_dataset1_to_bio(df1)
    sents2 = convert_dataset2_to_bio(df2)
    sents3 = convert_dataset3_to_bio(lines3)
    
    # Unify tags
    all_sentences = sents1 + sents2 + sents3
    print(f"\nTotal sentences before unification: {len(all_sentences)}")
    
    unified = unify_tags(all_sentences, "all")
    
    # Save full unified dataset
    save_bio_format(unified, "data/unified_bio.txt")
    
    # Create non-IID partitions
    client_data = create_non_iid_partitions(unified, num_clients=3)
    
    # Split train/test
    train_data, test_data = split_train_test(client_data)
    
    # Save per-client data
    for i in range(3):
        save_bio_format(train_data[i], f"data/client_{i}_train.txt")
        save_bio_format(test_data[i], f"data/client_{i}_test.txt")
    
    # Also save combined train/test for centralized baseline
    all_train = [s for client in train_data for s in client]
    all_test = [s for client in test_data for s in client]
    save_bio_format(all_train, "data/centralized_train.txt")
    save_bio_format(all_test, "data/centralized_test.txt")
    
    print("\nPreprocessing complete!")
    print("Files created in data/:")
    print("  - unified_bio.txt (all data)")
    print("  - client_0_train.txt, client_0_test.txt")
    print("  - client_1_train.txt, client_1_test.txt")
    print("  - client_2_train.txt, client_2_test.txt")
    print("  - centralized_train.txt, centralized_test.txt")


if __name__ == "__main__":
    main()