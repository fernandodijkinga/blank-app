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
                "content": "Você é um especialista em melhoramento genético animal e bovinos leiteiros. Responda sempre em português em formato de relatório técnico."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
        max_tokens=10000  # Limit the tokens to 5000
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

# Trait mapping with additional details
trait_mapping = {
    "Health": {"name": "Índice de Saúde", "definition": "Indica a saúde geral do animal", "average": 0, "unit": ""},
    "ProductionEfficiency": {"name": "Eficiência Alimentar", "definition": "Eficiência na conversão de alimento em produção", "average": 0, "unit": ""},
    "FertilityAndFitness": {"name": "Índice de Fertilidade", "definition": "Indica a capacidade de fertilidade do animal", "average": 0, "unit": ""},
    "LifetimeNetMerit": {"name": "Mérito Líquido", "definition": "Valor econômico do animal ao longo da vida", "average": 0, "unit": "USD"},
    "LifetimeCheeseMerit": {"name": "Mérito de Queijo", "definition": "Valor do animal para produção de queijo", "average": 0, "unit": "USD"},
    "TPI": {"name": "TPI", "definition": "Índice total de produção", "average": 2800, "unit": ""},
    "Milk": {"name": "Leite", "definition": "Quantidade de leite produzida", "average": 0, "unit": "lbs"},
    "Fat": {"name": "Gordura", "definition": "Quantidade de gordura no leite", "average": 0, "unit": "lbs"},
    "FatPercent": {"name": "Percentual de Gordura", "definition": "Percentual de gordura no leite", "average": 0, "unit": "%"},
    "Protein": {"name": "Proteína", "definition": "Quantidade de proteína no leite", "average": 0, "unit": "lbs"},
    "ProteinPercent": {"name": "Percentual de Proteína", "definition": "Percentual de proteína no leite", "average": 0, "unit": "%"},
    "CombinedFatAndProtein": {"name": "Gordura e Proteína Combinadas", "definition": "Quantidade total de gordura e proteína", "average": 0, "unit": "lbs"},
    "ProductiveLife": {"name": "Vida Produtiva", "definition": "Duração produtiva do animal", "average": 0, "unit": "meses"},
    "CowLivability": {"name": "Taxa de Sobrevivência da Vaca", "definition": "Probabilidade de sobrevivência da vaca", "average": 0, "unit": "%"},
    "SomaticCellScore": {"name": "Células Somáticas", "definition": "Contagem de células somáticas no leite", "average": 3, "unit": ""},
    "DaughterPregnancyRate": {"name": "DPR - Taxa de Prenhez", "definition": "Taxa de prenhez das filhas", "average": 0, "unit": "%"},
    "CowConceptionRate": {"name": "CCR - Taxa de Concepção de Vacas", "definition": "Taxa de concepção das vacas", "average": 0, "unit": "%"},
    "HeiferConceptionRate": {"name": "HCR - Taxa de Concepção de Novilhas", "definition": "Taxa de concepção das novilhas", "average": 0, "unit": "%"},
    "ExpectedFutureInbreeding": {"name": "Consanguinidade Esperada das Filhas", "definition": "Nível esperado de consanguinidade nas filhas", "average": 0, "unit": "%"},
    "CDCBHypocaicemia": {"name": "Hipocalcemia", "definition": "Incidência de hipocalcemia", "average": 0, "unit": "%"},
    "CDCBDisplacedAbomasum": {"name": "Deslocamento de Abomaso", "definition": "Incidência de deslocamento de abomaso", "average": 0, "unit": "%"},
    "CDCBKetosis": {"name": "Cetose", "definition": "Incidência de cetose", "average": 0, "unit": "%"},
    "CDCBMastitis": {"name": "Mastite", "definition": "Incidência de mastite", "average": 0, "unit": "%"},
    "CDCBMetritis": {"name": "Metrite", "definition": "Incidência de metrite", "average": 0, "unit": "%"},
    "CDCBRetainedPlacenta": {"name": "Retenção de Placenta", "definition": "Incidência de retenção de placenta", "average": 0, "unit": "%"},
    "BodyConditionScore": {"name": "Escore de Condição Corporal", "definition": "Avaliação da condição corporal do animal", "average": 0, "unit": ""},
    "MilkingSpeed": {"name": "Velocidade de Ordenha", "definition": "Velocidade de ordenha do animal", "average": 0, "unit": ""},
    "MilkingTemperament": {"name": "Temperamento na Ordenha", "definition": "Comportamento do animal durante a ordenha", "average": 0, "unit": ""},
    "SireCalvingEase": {"name": "Facilidade de Parto", "definition": "Facilidade de parto proporcionada pelo touro", "average": 0, "unit": "%"},
    "SireStillbirth": {"name": "Natimortos", "definition": "Taxa de natimortos", "average": 0, "unit": "%"},
    "PredictedTransmittingAbilityType": {"name": "PTAT", "definition": "Capacidade de transmissão prevista", "average": 0, "unit": ""},
    "UdderComposite": {"name": "Composto de Úbere", "definition": "Avaliação geral do úbere", "average": 0, "unit": ""},
    "FeetAndLegComposite": {"name": "Composto de Pernas e Pés", "definition": "Avaliação geral das pernas e pés", "average": 0, "unit": ""},
    "BodySizeComposite": {"name": "Composto Corporal", "definition": "Avaliação geral do corpo", "average": 0, "unit": ""},
    "DairyComposite": {"name": "Composto Leiteiro", "definition": "Avaliação geral da capacidade leiteira", "average": 0, "unit": ""},
    "Stature": {"name": "Estatura", "definition": "Altura do animal", "average": 0, "unit": "cm"},
    "Strength": {"name": "Força", "definition": "Força física do animal", "average": 0, "unit": ""},
    "BodyDepth": {"name": "Profundidade Corporal", "definition": "Profundidade do corpo do animal", "average": 0, "unit": "cm"},
    "DairyForm": {"name": "Forma Leiteira", "definition": "Conformação física voltada para a produção de leite", "average": 0, "unit": ""},
    "RumpAngle": {"name": "Ângulo de Garupa", "definition": "Ângulo da garupa do animal", "average": 0, "unit": "graus"},
    "ThurlWidth": {"name": "Largura de Garupa", "definition": "Largura da garupa do animal", "average": 0, "unit": "cm"},
    "RearLegsSideView": {"name": "Pernas Vista Lateral", "definition": "Conformação das pernas vistas de lado", "average": 0, "unit": ""},
    "RearLegsRearView": {"name": "Pernas Vista Posterior", "definition": "Conformação das pernas vistas por trás", "average": 0, "unit": ""},
    "FootAngle": {"name": "Ângulo de Casco", "definition": "Ângulo do casco do animal", "average": 0, "unit": "graus"},
    "FeetAndLegsScore": {"name": "Escore de Pernas e Pés", "definition": "Avaliação das pernas e pés do animal", "average": 0, "unit": ""},
    "ForeUdderAttachment": {"name": "Inserção de Úbere Anterior", "definition": "Avaliação da inserção do úbere anterior", "average": 0, "unit": ""},
    "RearUdderHeight": {"name": "Altura de Úbere Posterior", "definition": "Avaliação da altura do úbere posterior", "average": 0, "unit": "cm"},
    "RearUdderWidth": {"name": "Largura de Úbere Posterior", "definition": "Avaliação da largura do úbere posterior", "average": 0, "unit": "cm"},
    "UdderDepth": {"name": "Ligamento Central", "definition": "Avaliação do ligamento central do úbere", "average": 0, "unit": ""},
    "UdderCleft": {"name": "Profundidade de Úbere Posterior", "definition": "Avaliação da profundidade do úbere posterior", "average": 0, "unit": "cm"},
    "FrontTeatPlacement": {"name": "Colocação de Tetos Anteriores", "definition": "Avaliação da colocação dos tetos anteriores", "average": 0, "unit": ""},
    "RearTeatPlacement": {"name": "Colocação de Tetos Posteriores", "definition": "Avaliação da colocação dos tetos posteriores", "average": 0, "unit": ""},
    "TeatLength": {"name": "Comprimento de Tetos", "definition": "Comprimento dos tetos", "average": 0, "unit": "cm"},
    "BetaCasein": {"name": "Beta Caseína", "definition": "Presença de beta caseína no leite", "average": 0, "unit": ""},
    "KappaCasein": {"name": "Kappa Caseína", "definition": "Presença de kappa caseína no leite", "average": 0, "unit": ""},
    "FeedSaved": {"name": "Feed Saved", "definition": "Quantidade de alimento economizada", "average": 0, "unit": "lbs"},
    "HeiferLivability": {"name": "Taxa de Sobrevivência de Novilhas", "definition": "Taxa de sobrevivência das novilhas", "average": 0, "unit": "%"},
    "DurationPregnancy": {"name": "Duração da Gestação", "definition": "Duração da gestação", "average": 0, "unit": "dias"},
    "AgeFirstChildbirth": {"name": "Idade ao Primeiro Parto", "definition": "Idade ao primeiro parto", "average": 0, "unit": "meses"}
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
def gerar_prompt_touros(touros, traits_df, task, trait_references):
    prompt = f"Task: {task}\n"
    prompt += f"Trait References:\n{trait_references}\n\n"
    for _, touro in touros.iterrows():
        prompt += "**Informações do touro**\n"
        prompt += f"Apelido: {touro['ShortName']}\n"
        prompt += f"Nome: {touro['LongName']}\n"
        prompt += f"NAAB: {touro['NAABCode']}\n"
        prompt += f"Registro: {touro[touros_key_column]}\n\n"
        
        prompt += "**Índices**\n"
        for trait in index_traits:
            if trait in touro:
                prompt += f"{trait_mapping[trait]['name']}: {touro[trait]}\n"
        
        prompt += "\n**Características de produção e eficiência**\n"
        for trait in production_traits:
            if trait in touro:
                prompt += f"{trait_mapping[trait]['name']}: {touro[trait]}\n"
        
        prompt += "\n**Longevidade e saúde**\n"
        for trait in health_traits:
            if trait in touro:
                prompt += f"{trait_mapping[trait]['name']}: {touro[trait]}\n"
        
        prompt += "\n**Fertilidade e parto**\n"
        for trait in fertility_traits:
            if trait in touro:
                prompt += f"{trait_mapping[trait]['name']}: {touro[trait]}\n"
        
        prompt += "\n**Compostos**\n"
        for trait in compounds_traits:
            if trait in touro:
                prompt += f"{trait_mapping[trait]['name']}: {touro[trait]}\n"

        prompt += "\n\n"
    return prompt

# Function to display formatted bull information with smaller font
def exibir_informacoes_touro(touro):
    st.markdown("<div style='font-size: small;'>", unsafe_allow_html=True)
    st.write(gerar_prompt_touros(pd.DataFrame([touro]), traits_df, "", ""))
    st.markdown("</div>", unsafe_allow_html=True)

# Function to search for bull traits
def consulta_touro(touros_df, search_term):
    touro = touros_df[touros_df['LongName'] == search_term]
    if touro.empty:
        st.write("Touro não encontrado.")
    else:
        exibir_informacoes_touro(touro.iloc[0])
        task = "Consulta de Prova de Touro"
        trait_references = "\n".join([f"{trait_mapping[trait]['name']}: {trait_mapping[trait]['definition']}, Média: {trait_mapping[trait]['average']}, Unidade: {trait_mapping[trait]['unit']}" for trait in index_traits + production_traits + health_traits + fertility_traits + compounds_traits + conformation_traits])
        response = generate_response(gerar_prompt_touros(touro, traits_df, task, trait_references))
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
        
        task = "Comparar Touros"
        trait_references = "\n".join([f"{trait_mapping[trait]['name']}: {trait_mapping[trait]['definition']}, Média: {trait_mapping[trait]['average']}, Unidade: {trait_mapping[trait]['unit']}" for trait in index_traits + production_traits + health_traits + fertility_traits + compounds_traits + conformation_traits])
        response = generate_response(gerar_prompt_touros(touros, traits_df, task, trait_references))
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

        task = "Encontrar Touros Similares"
        trait_references = "\n".join([f"{trait_mapping[trait]['name']}: {trait_mapping[trait]['definition']}, Média: {trait_mapping[trait]['average']}, Unidade: {trait_mapping[trait]['unit']}" for trait in trait_group])
        response = generate_response(gerar_prompt_touros(pd.concat([touro, similar_bulls]), traits_df, task, trait_references))
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
