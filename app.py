import streamlit as st
import pandas as pd
import os
import torch
import kagglehub
import json
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Contendipity", layout="wide", page_icon="🌌")

@st.cache_resource
def inizializza_sistema():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)
    
    # video (film o serie tv)
    video_files = ["netflix_titles.csv", "disney_plus_titles.csv", "hulu_titles.csv", "amazon_prime_titles.csv"]
    all_v = []
    for f in video_files:
        if os.path.exists(f):
            df = pd.read_csv(f, usecols=['title', 'description'])
            df['platform'] = f.split('_')[0].capitalize()
            all_v.append(df)
    
    if not all_v:
        video_df = pd.DataFrame(columns=['title', 'description', 'platform'])
    else:
        video_df = pd.concat(all_v, ignore_index=True).dropna().reset_index(drop=True)

    # libri
    path_b = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
    books_df = pd.read_csv(os.path.join(path_b, "books_data.csv"), nrows=10000, usecols=['Title', 'description'])
    books_df = books_df.dropna().reset_index(drop=True) # esclude i record se il libro se non ha descrizione (NaN)

    # podcast
    path_p = kagglehub.dataset_download("thoughtvector/podcastreviews")
    file_json = next((os.path.join(r, f) for r, d, fs in os.walk(path_p) for f in fs if f == "podcasts.json"), None)
    pod_list = []
    if file_json:
        with open(file_json, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000: break
                try:
                    data = json.loads(line)
                    pod_list.append({
                        'title': data.get('title', data.get('name', '')),
                        'description': data.get('description', data.get('notes', ''))
                    })
                except: continue
    podcasts_df = pd.DataFrame(pod_list).dropna().reset_index(drop=True)

    # mappe semantiche (vettori)
    v_emb = model.encode((video_df['title'] + " " + video_df['description']).tolist(), convert_to_tensor=True) if not video_df.empty else None
    b_emb = model.encode((books_df['Title'] + " " + books_df['description']).tolist(), convert_to_tensor=True)
    p_emb = model.encode((podcasts_df['title'] + " " + podcasts_df['description']).tolist(), convert_to_tensor=True)

    return model, video_df, books_df, podcasts_df, v_emb, b_emb, p_emb

# avvio interfaccia
with st.spinner("🌌 Contendipity sta analizzando l'Universo..."):
    model, video_df, books_df, podcasts_df, v_emb, b_emb, p_emb = inizializza_sistema()

st.title("🌌 Contendipity")
st.markdown("*Per quando ami così tanto qualcosa da non voler più uscire da quel mondo*")

# ricerca
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Inserisci un titolo che ti è piaciuto:", placeholder="Es: Modern Family, Friends, Brooklyn Nine-Nine...")
with col2:
    tipo = st.selectbox("In quale categoria si trova?", ["video", "libro", "podcast"])

if query:
    # selezione del database di partenza
    if tipo == "video": t_df, t_emb, t_col = video_df, v_emb, "title"
    elif tipo == "libro": t_df, t_emb, t_col = books_df, b_emb, "Title"
    else: t_df, t_emb, t_col = podcasts_df, p_emb, "title"

    # logica di ricerca (Match Esatto > Contiene)
    match_esatto = t_df[t_df[t_col].str.lower() == query.lower()]
    if not match_esatto.empty:
        idx = match_esatto.index[0]
    else:
        matches = t_df[t_df[t_col].str.contains(query, case=False, na=False)]
        if matches.empty:
            st.error(f"Impossibile trovare '{query}' nella categoria {tipo}.")
            st.stop()
        idx = matches.assign(l=matches[t_col].str.len()).sort_values('l').index[0]

    v_query = t_emb[idx]
    titolo_origine = t_df.iloc[idx][t_col]
    
    st.info(f"Ho trovato: **{titolo_origine}**. Ecco cosa potrebbe piacerti:")
    
    # risultati
    tabs = st.tabs(["🎬 Video Correlati", "📚 Libri Simili", "🎙️ Podcast Affini"])
    
    categorie = [
        ("Video", video_df, v_emb, tabs[0]),
        ("Libri", books_df, b_emb, tabs[1]),
        ("Podcast", podcasts_df, p_emb, tabs[2])
    ]

    for nome_cat, df_d, emb_d, tab in categorie:
        with tab:
            if emb_d is not None:
                # calcolo della similarità
                scores = util.cos_sim(v_query, emb_d)[0]
                top_k = torch.topk(scores, k=6)
                
                for s, i in zip(top_k.values, top_k.indices):
                    res = df_d.iloc[i.item()]
                    t_res = res['Title'] if 'Title' in res else res['title']
                    
                    # escludi il titolo inserito dall'utente
                    if t_res.lower() == titolo_origine.lower():
                        continue
                    
                    with st.expander(f"⭐ {t_res} ({int(s.item()*100)}% affine)"):
                        st.write(res['description'])
                        if 'platform' in res:
                            st.caption(f"Disponibile su: {res['platform']}")