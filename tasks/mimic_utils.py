import argparse

import pandas as pd
import os
import re
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--admission_only', default=False)
    parser.add_argument('--seed', default=123, type=int)

    return parser.parse_args()


def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def _norm_header(s: str) -> str:
    # normalize header for matching: lowercase, collapse spaces, remove trailing colon
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s[:-1] if s.endswith(":") else s
    return s

def build_header_map(admission_sections: dict[str, str], extra_variants: dict[str, list[str]] | None = None):
    """
    admission_sections: {CANON_KEY: "Chief Complaint:"}
    extra_variants: {CANON_KEY: ["CC:", "CHIEF COMPLAINT", ...]}
    returns: normalized_header -> CANON_KEY
    """
    header_map = {}
    for canon_key, header in admission_sections.items():
        header_map[_norm_header(header)] = canon_key
        header_map[_norm_header(header.rstrip(":"))] = canon_key

    if extra_variants:
        for canon_key, variants in extra_variants.items():
            for v in variants:
                header_map[_norm_header(v)] = canon_key
                header_map[_norm_header(v.rstrip(":"))] = canon_key

    return header_map

def extract_sections(note_text: str, header_map: dict[str, str]) -> dict[str, str]:
    """
    Find headers as standalone lines, slice content between successive headers.
    """
    header_line_re = re.compile(r"(?m)^(?P<h>[A-Za-z][A-Za-z0-9 /&\-\(\)\[\]]{0,60}?):?\s*$")

    matches = []
    for m in header_line_re.finditer(note_text):
        raw = m.group("h")
        canon = header_map.get(_norm_header(raw))
        if canon:
            matches.append((canon, m.start(), m.end()))  # start/end of header line

    if not matches:
        return {}

    # De-dupe: if same canon appears multiple times, you can choose first, last, or concat.
    # Besides: We'll concat with "\n\n" in order to stick to the original structure.
    out = {}
    for idx, (canon, h_start, h_end) in enumerate(matches):
        content_start = h_end
        content_end = matches[idx + 1][1] if idx + 1 < len(matches) else len(note_text)
        content = note_text[content_start:content_end].strip()

        if canon in out and content:
            out[canon] = (out[canon].rstrip() + "\n\n" + content)
        elif content:
            out[canon] = content
        else:
            out.setdefault(canon, "")

    return out


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the sections of the discharge summary that are known at admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "Chief Complaint:",
        "PRESENT_ILLNESS": "History of Present Illness:",
        "MEDICAL_HISTORY": "Past Medical History:",
        "MEDICATION_ADM": "Medications on Admission:",
        "ALLERGIES": "Allergies:",
        "PHYSICAL_EXAM": "Physical Exam:",
        "FAMILY_HISTORY": "Family History:",
        "SOCIAL_HISTORY": "Social History:"
    }

    extra_variants = {
        "PRESENT_ILLNESS": ["Present Illness", "HPI", "History Present Illness"],
        "MEDICAL_HISTORY": ["PMH", "Medical History"],
        "MEDICATION_ADM": ["Home Medications", "Medications"],
    }

    header_map = build_header_map(admission_sections, extra_variants)

    for key in admission_sections.keys():
        notes_df[key] = ""

    for i, x in enumerate(notes_df["TEXT"]):
        sec = extract_sections(x, header_map)
        for k, v in sec.items():
            notes_df.at[i, k] = v

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df


def save_mimic_split_patient_wise(df, label_column, save_dir, task_name, seed, column_list=None):
    """
    Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "TEXT", label_column]

    # Load prebuilt MIMIC patient splits
    data_split = {"train": pd.read_csv("tasks/mimic_train.csv"),
                  "val": pd.read_csv("tasks/mimic_val.csv"),
                  "test": pd.read_csv("tasks/mimic_test.csv")}

    # Use row id as general id and cast to int
    df = df.rename(columns={'HADM_ID': 'ID'})
    df.ID = df.ID.astype(int)

    # Create path to task data
    os.makedirs(save_dir, exist_ok=True)

    # Save splits to data folder
    for split_name in ["train", "val", "test"]:
        split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                     random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)

        split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)
