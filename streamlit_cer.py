import streamlit as st
import pandas as pd
from torchmetrics.text import CharErrorRate
from text_normalize import normalize_sentence

st.title("Character Error Rate (CER) Evaluation")

# Load xlsx file
uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Dataframe:")
    st.dataframe(df)


    # Select which column has the reference and which has the transcription
    columns = df.columns.tolist()
    reference_col = st.selectbox("Select the reference column", columns)
    transcription_col = st.selectbox("Select the transcription column", columns)
    
    result_df = pd.DataFrame(columns=["reference", "transcription", "cer"])
    cer = CharErrorRate()
    for idx, row in df.iterrows():
        reference_str = str(row[reference_col])
        transcription_str = str(row[transcription_col])

        reference =  ''.join(normalize_sentence(reference_str))
        transcription =  ''.join(normalize_sentence(transcription_str))

        cer_score = cer(transcription, reference)
        
        result_df.loc[idx] = [reference_str, transcription_str, cer_score.item()]
        
        
    st.write("CER Results:")
    st.dataframe(result_df)