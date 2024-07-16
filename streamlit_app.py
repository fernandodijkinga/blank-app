import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from groq import Groq

# Initialize Groq client
client = Groq(api_key="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")

# Function to generate LLM response
def generate_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Você é um especialista em melhoramento genético animal e bovinos leiteiros. Responda sempre em português."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
        max_tokens=5000  # Limit the tokens to 5000
    )
    response = chat_completion.choices[0].message.content
    return response

# Function to normalize data
def normalize(df, columns):
    available_columns = [col for col in columns if col in df.columns]
    return (df[available_columns] - df[available_columns].mean()) / df[available_columns].std()

# Load CSV files
touros_df = pd.read_csv('.github/Touros202404.csv', delimiter=";")
traits_file_path = '.github/traits.csv'
traits_df = pd.read_csv(traits_file_path, delimiter=";")

# Confirm column names
touros_key_column = 'BullKey' if 'BullKey' in touros_df.columns else 'BullCode'

# Trait mapping
trait_mapping = {
    "Health": "Índice de Saúde",
    "ProductionEfficiency": "Eficiência Alimentar",
    "FertilityAndFitness": "Índice de Fertilidade",
    "LifetimeNetMerit": "Mérito Líquido",
    "LifetimeCheeseMerit": "Mérito de Queijo",
    "TPI": "TPI",
    "Milk": "Leite",
    "Fat": "Gordura",
    "FatPercent": "Percentual de Gordura",
    "Protein": "Proteína",
    "ProteinPercent": "Percentual de Proteína",
    "CombinedFatAndProtein": "Gordura e Proteína Combinadas",
    "ProductiveLife": "Vida Produtiva",
    "CowLivability": "Taxa de Sobrevivência da Vaca",
    "SomaticCellScore": "Células Somáticas",
    "DaughterPregnancyRate": "DPR - Taxa de Prenhez",
    "CowConceptionRate": "CCR - Taxa de Concepção de Vacas",
    "HeiferConceptionRate": "HCR - Taxa de Concepção de Novilhas",
    "ExpectedFutureInbreeding": "Consanguinidade Esperada das Filhas",
    "CDCBHypocaicemia": "Hipocalcemia",
    "CDCBDisplacedAbomasum": "Deslocamento de Abomaso",
    "CDCBKetosis": "Cetose",
    "CDCBMastitis": "Mastite",
    "CDCBMetritis": "Metrite",
    "CDCBRetainedPlacenta": "Retenção de Placenta",
    "BodyConditionScore": "Escore de Condição Corporal",
    "MilkingSpeed": "Velocidade de Ordenha",
    "MilkingTemperament": "Temperamento na Ordenha",
    "SireCalvingEase": "Facilidade de Parto",
    "SireStillbirth": "Natimortos",
    "PredictedTransmittingAbilityType": "PTAT",
    "UdderComposite": "Composto de Úbere",
    "FeetAndLegComposite": "Composto de Pernas e Pés",
    "BodySizeComposite": "Composto Corporal",
    "DairyComposite": "Composto Leiteiro",
    "Stature": "Estatura",
    "Strength": "Força",
    "BodyDepth": "Profundidade Corporal",
    "DairyForm": "Forma Leiteira",
    "RumpAngle": "Ângulo de Garupa",
    "ThurlWidth": "Largura de Garupa",
    "RearLegsSideView": "Pernas Vista Lateral",
    "RearLegsRearView": "Pernas Vista Posterior",
    "FootAngle": "Ângulo de Casco",
    "FeetAndLegsScore": "Escore de Pernas e Pés",
    "ForeUdderAttachment": "Inserção de Úbere Anterior",
    "RearUdderHeight": "Altura de Úbere Posterior",
    "RearUdderWidth": "Largura de Úbere Posterior",
    "UdderDepth": "Ligamento Central",
    "UdderCleft": "Profundidade de Úbere Posterior",
    "FrontTeatPlacement": "Colocação de Tetos Anteriores",
    "RearTeatPlacement": "Colocação de Tetos Posteriores",
    "TeatLength": "Comprimento de Tetos",
    "BetaCasein": "Beta Caseína",
    "KappaCasein": "Kappa Caseína",
    "FeedSaved": "Feed Saved",
    "HeiferLivability": "Taxa de Sobrevivência de Novilhas",
    "DurationPregnancy": "Duração da Gestação",
    "AgeFirstChildbirth": "Idade ao Primeiro Parto"
}

# Trait groups
main_traits = ["TPI", "LifetimeNetMerit", "LifetimeCheeseMerit", "Milk", "Fat", "Protein", "ProductionEfficiency", "ProductiveLife", "CowLivability", "SomaticCellScore", "Health", "DaughterPregnancyRate", "PredictedTransmittingAbilityType", "UdderComposite", "FeetAndLegComposite"]
index_traits = ["TPI", "LifetimeNetMerit", "LifetimeCheeseMerit"]
production_traits = ["Milk", "Fat", "FatPercent", "Protein", "ProteinPercent", "ProductionEfficiency"]
health_traits = ["ProductiveLife", "CowLivability", "SomaticCellScore", "Health"]
fertility_traits = ["DaughterPregnancyRate", "CowConceptionRate", "HeiferConceptionRate", "FertilityAndFitness"]
compounds_traits = ["PredictedTransmittingAbilityType", "UdderComposite", "FeetAndLegComposite", "BodySizeComposite", "DairyComposite"]
conformation_traits = ["Stature", "Strength", "BodyDepth", "DairyForm", "RumpAngle", "ThurlWidth", "RearLegsSideView", "RearLegsRearView", "FootAngle", "FeetAndLegsScore", "ForeUdderAttachment", "RearUdderHeight", "RearUdderWidth", "UdderDepth", "UdderCleft", "FrontTeatPlacement", "RearTeatPlacement", "TeatLength"]

# Function to generate formatted prompt for multiple bulls
def gerar_prompt_touros(touros, traits_df):
    prompt = ""
    for _, touro in touros.iterrows():
        prompt += "**Informações do touro**\n"
        prompt += f"Apelido: {touro['ShortName']}\n"
        prompt += f"Nome: {touro['LongName']}\n"
        prompt += f"NAAB: {touro['NAABCode']}\n"
        prompt += f"Registro: {touro[touros_key_column]}\n\n"
        
        prompt += "**Índices**\n"
        for trait in index_traits:
            if trait in touro:
                prompt += f"\n{trait_mapping[trait]}: {touro[trait]}\n"
        
        prompt += "\n**Características de produção e eficiência**\n"
        for trait in production_traits:
            if trait in touro:
                prompt += f"\n{trait_mapping[trait]}: {touro[trait]}\n"
        
        prompt += "\n**Longevidade e saúde**\n"
        for trait in health_traits:
            if trait in touro:
                prompt += f"\n{trait_mapping[trait]}: {touro[trait]}\n"
        
        prompt += "\n**Fertilidade e parto**\n"
        for trait in fertility_traits:
            if trait in touro:
                prompt += f"\n{trait_mapping[trait]}: {touro[trait]}\n"
        
        prompt += "\n**Compostos**\n"
        for trait in compounds_traits:
            if trait in touro:
                prompt += f"\n{trait_mapping[trait]}: {touro[trait]}\n"

        prompt += "\n\n"

    return prompt

# Function to display formatted bull information with smaller font
def exibir_informacoes_touro(touro):
    st.markdown("<div style='font-size: small;'>", unsafe_allow_html=True)
    st.write(gerar_prompt_touros(pd.DataFrame([touro]), traits_df))
    st.markdown("</div>", unsafe_allow_html=True)

# Function to search for bull traits
def consulta_touro(touros_df, search_term):
    touro = touros_df[touros_df['LongName'] == search_term]
    if touro.empty:
        st.write("Touro não encontrado.")
    else:
        exibir_informacoes_touro(touro.iloc[0])
        response = generate_response(gerar_prompt_touros(touro, traits_df))
        st.write("\n**Resposta da LLM**\n")
        st.write(response)

# Function to compare bulls
def comparar_touros(touros_df, search_terms):
    touros = touros_df[touros_df['LongName'].isin(search_terms)]
    if touros.empty:
        st.write("Nenhum touro encontrado.")
    else:
        available_columns = [col for col in ["ShortName", "LongName", "NAABCode", touros_key_column] + index_traits + production_traits + health_traits + fertility_traits + compounds_traits + conformation_traits if col in touros.columns]
        comparison_df = touros[available_columns]
        st.dataframe(comparison_df)
        
        response = generate_response(gerar_prompt_touros(touros, traits_df))
        st.write("\n**Resposta da LLM**\n")
        st.write(response)

# Function to find similar bulls
def touros_similares(touros_df, search_term, trait_group, companies=None):
    touro = touros_df[touros_df['LongName'] == search_term]
    if touro.empty:
        st.write("Touro não encontrado.")
    else:
        exibir_informacoes_touro(touro.iloc[0])
        trait_columns = [col for col in trait_group if col in touros_df.columns]
        normalized_traits = normalize(touros_df, trait_columns)
        reference_traits = normalized_traits[touros_df[touros_key_column] == touro[touros_key_column].values[0]].values.flatten()
        
        if companies:
            touros_df_filtered = touros_df[touros_df['Company'].isin(companies)]
        else:
            touros_df_filtered = touros_df.copy()
        
        touros_df_filtered = touros_df_filtered[touros_df_filtered[touros_key_column] != touro[touros_key_column].values[0]]
        
        normalized_traits_filtered = normalized_traits.loc[touros_df_filtered.index]
        
        touros_df_filtered['distance'] = normalized_traits_filtered.apply(lambda row: euclidean(reference_traits, row), axis=1)
        similar_bulls = touros_df_filtered.sort_values(by='distance').head(5)
        
        st.write("Touros Similares:")
        available_columns = [col for col in ["ShortName", "LongName", "NAABCode", touros_key_column] + trait_columns if col in similar_bulls.columns]
        comparison_df = similar_bulls[available_columns]
        st.dataframe(comparison_df)

        response = generate_response(gerar_prompt_touros(similar_bulls, traits_df))
        st.write("\n**Resposta da LLM**\n")
        st.write(response)

# Streamlit Interface
st.title("Consulta de Touros")

funcionalidade = st.selectbox("Escolha uma funcionalidade", ["Consulta de Prova de Touro", "Comparar Touros", "Touro Similar"])

if funcionalidade == "Consulta de Prova de Touro":
    search_term = st.selectbox("Digite o LongName do touro", touros_df['LongName'].tolist())
    if st.button("Buscar"):
        consulta_touro(touros_df, search_term)
elif funcionalidade == "Comparar Touros":
    search_terms = st.multiselect("Digite os LongNames dos touros", touros_df['LongName'].tolist())
    if st.button("Comparar"):
        comparar_touros(touros_df, search_terms)
elif funcionalidade == "Touro Similar":
    search_term = st.selectbox("Digite o LongName do touro de referência", touros_df['LongName'].tolist())
    trait_group_name = st.selectbox("Escolha o grupo de características", ["Principais características", "Produção", "Saúde e fertilidade", "Conformação"])
    trait_group = []
    if trait_group_name == "Principais características":
        trait_group = main_traits
    elif trait_group_name == "Produção":
        trait_group = production_traits
    elif trait_group_name == "Saúde e fertilidade":
        trait_group = health_traits
    elif trait_group_name == "Conformação":
        trait_group = conformation_traits
    
    companies = st.multiselect("Escolha as empresas (opcional)", touros_df['Company'].unique().tolist())
    
    if st.button("Encontrar Touros Similares"):
        touros_similares(touros_df, search_term, trait_group, companies)
