import streamlit as st
import pandas as pd
import re
from groq import Groq
from scipy.spatial.distance import euclidean

idioma = "portugues"

# Initialize Groq client
client = Groq(api_key="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")

# Load the CSV files
file_path = '/workspaces/blank-app/Touros202404.csv'
touros_df = pd.read_csv(file_path, delimiter=";")
traits_file_path = '/workspaces/blank-app/traits.csv'
traits_df = pd.read_csv(traits_file_path, delimiter=";")

# Ensure all values in 'ShortName', 'NAABCode', 'BullKey', 'LongName', and 'Company' are treated as strings
touros_df['ShortName'] = touros_df['ShortName'].astype(str)
touros_df['NAABCode'] = touros_df['NAABCode'].astype(str)
touros_df['BullKey'] = touros_df['BullKey'].astype(str)
touros_df['LongName'] = touros_df['LongName'].astype(str)
touros_df['Company'] = touros_df['Company'].astype(str)

# Print columns for debugging
print("Columns in traits_df:", traits_df.columns)

# Trait mapping (example, you can expand this as needed)
trait_mapping = {
    "Health": ["indice de saude", "filhas mais saudaveis"],
    "ProductionEfficiency": "eficiencia alimentar",
    "FertilityAndFitness": "indice de fertilidade",
    "LifetimeNetMerit": "Merito Liquido",
    "LifetimeCheeseMerit": "Merito de Queijo",
    "TPI": "TPI",
    "Milk": "Leite",
    "Fat": "Gordura",
    "FatPercent": "Percentual de Gordura",
    "Protein": "Proteina",
    "ProteinPercent": "Percentual de Proteina",
    "CombinedFatAndProtein": "Gordura e Proteína Combinadas",
    "ProductiveLife": "Vida Produtiva",
    "CowLivability": "Viabilidade da Vaca",
    "SomaticCellScore": "Células Somáticas",
    "DaughterPregnancyRate": ["DPR", "Taxa de prenhez", "dpr"],
    "CowConceptionRate": ["CCR", "Taxa de concepcao de vacas"],
    "HeiferConceptionRate": ["HCR", "Taxa de concepcao de novilhas"],
    "ExpectedFutureInbreeding": ["Consanguinidade", "Endogamia", "consanguinidade futura", "consanguinidade esperada", "consanguinidade esperada das filhas"],
    "CDCBHypocaicemia": ["Febre do leite", "Hipocalcemia"],
    "CDCBDisplacedAbomasum": "Deslocamento de abomaso",
    "CDCBKetosis": "Cetose",
    "CDCBMastitis": "Mastite",
    "CDCBMetritis": "Metrite",
    "CDCBRetainedPlacenta": "Retencao de placenta",
    "BodyConditionScore": "Escore de condicao corporal",
    "MilkingSpeed": "Velocidade de ordenha",
    "MilkingTemperament": "Temperamento na ordenha",
    "SireCalvingEase": "facilidade de parto",
    "SireStillbirth": "natimortos",
    "PredictedTransmittingAbilityType": ["PTAT", "pontuacao final", "pta tipo", "conformacao"],
    "UdderComposite": ["Composto de ubere", "ubere mais alto e largo"],
    "FeetAndLegComposite": "composto de pernas e pés",
    "BodySizeComposite": "composto corporal",
    "DairyComposite": "composto leiteiro",
    "Stature": ["estatura", "altura"],
    "Strength": "força",
    "BodyDepth": "profundidade corporal",
    "DairyForm": ["forma leiteira", "angulosidade"],
    "RumpAngle": "angulo de garupa",
    "ThurlWidth": "largura de garupa",
    "RearLegsSideView": "pernas vista lateral",
    "RearLegsRearView": "pernas vista posterior",
    "FootAngle": "angulo de casco",
    "FeetAndLegsScore": ["escore de pernas e pés", "locomocao"],
    "ForeUdderAttachment": ["insercao de ubere anterior", "ubere anterior", "fore udder attachment"],
    "RearUdderHeight": ["altura de ubere posterior", "ubere mais alto"],
    "RearUdderWidth": ["largura de ubere posterior", "ubere mais largo"],
    "UdderDepth": ["ligamento central", "ligamento suspensorio"],
    "UdderCleft": "profundidade de ubere posterior",
    "FrontTeatPlacement": "colocacao de tetos anteriores",
    "RearTeatPlacement": "colocacao de tetos posteriores",
    "TeatLength": "comprimento de tetos",
    "BetaCasein": "Beta caseina",
    "KappaCasein": "Kappa caseina",
    "FeedSaved": "feed saved",
    "HeiferLivability": ["taxa de sobrevivencia de novilhas", "heifer livability"],
    "DurationPregnancy": "duracao da gestacao",
    "AgeFirstChildbirth": "idade ao primeiro parto",
}

# Flatten the trait mapping for easier keyword detection in user query
keyword_to_trait = {}
for trait, keywords in trait_mapping.items():
    if isinstance(keywords, list):
        for keyword in keywords:
            keyword_to_trait[keyword.lower()] = trait
    else:
        keyword_to_trait[keywords.lower()] = trait

# Function to search for the specific bulls and retrieve PTA values for multiple traits
def get_bull_pta(dataframe, traits):
    pta_data = {}
    for index, row in dataframe.iterrows():
        bull_name = row['ShortName']
        pta_values = {}
        for trait in traits:
            if trait in row:
                pta_values[trait] = row[trait]
            else:
                pta_values[trait] = "Trait not found"
        pta_data[bull_name] = pta_values
    return pta_data

# Function to extract relevant traits from the user's query
def extract_relevant_traits(query):
    query_lower = query.lower()
    relevant_traits = [keyword_to_trait[keyword] for keyword in keyword_to_trait if keyword in query_lower and keyword_to_trait[keyword] != 'Company']
    return relevant_traits if relevant_traits else ["TPI", "Milk", "Fat", "FatPercent", "Protein", "ProteinPercent", "ProductionEfficiency", "ProductiveLife", "SomaticCellScore", "DaughterPregnancyRate", "Health", "PredictedTransmittingAbilityType", "UdderComposite", "FeetAndLegComposite"]  # Default to common traits if no specific traits are mentioned

# Function to extract company names from the user's query
def extract_company_names(query, known_companies):
    company_names = []
    for company in known_companies:
        if re.search(r'\b' + re.escape(company) + r'\b', query, re.IGNORECASE):
            company_names.append(company)
    return company_names

# Function to extract bull names from the user's query
def extract_bull_names(query, known_bulls):
    bull_names = []
    for bull in known_bulls:
        if isinstance(bull, float):  # Ensure bull is a string
            bull = str(bull)
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(bull) + r'\b', query, re.IGNORECASE):
            bull_names.append(bull)
    return bull_names

# Function to find trait definition in the CSV file
def find_trait_definition(traits, traits_df):
    trait_definitions = []
    for trait in traits:
        print(f"Searching definition for trait: {trait}")  # Debug print
        trait_lower = trait.lower()
        matching_traits = traits_df[traits_df['trait'].str.lower().str.contains(trait_lower)]
        if not matching_traits.empty:
            for _, row in matching_traits.iterrows():
                definition = row['definition']
                min_val = row['min']
                max_val = row['max']
                nature = row['nature']
                desired_value = row['desired_value']
                trait_definitions.append(f"Traço: {row['trait']}\nDefinição: {definition}\nMin: {min_val}\nMax: {max_val}\nNatureza: {nature}\nValor desejado: {desired_value}\n")
        else:
            trait_definitions.append(f"No definition found for trait: {trait}")  # Debug print
    return trait_definitions

# Function to find a similar bull from a specified company
def find_similar_bull(bull_name, target_company, traits, dataframe):
    # Get the PTA values of the specified bull
    print(f"Finding similar bull for: {bull_name} in company: {target_company} using traits: {traits}")  # Debug print
    specified_bull_pta = dataframe[
        (dataframe['ShortName'].str.lower() == bull_name.lower()) |
        (dataframe['NAABCode'].str.lower() == bull_name.lower()) |
        (dataframe['BullKey'].str.lower() == bull_name.lower()) |
        (dataframe['LongName'].str.lower() == bull_name.lower())
    ]
    if specified_bull_pta.empty:
        return f"Touro {bull_name} não encontrado."

    specified_bull_pta = specified_bull_pta.iloc[0]
    print(f"PTA values for {bull_name}: {specified_bull_pta[traits]}")  # Debug print
    specified_pta_values = specified_bull_pta[traits].apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Processed PTA values for {bull_name}: {specified_pta_values}")  # Debug print

    # Filter bulls from the target company
    company_bulls = dataframe[dataframe['Company'].str.lower() == target_company.lower()]
    if company_bulls.empty:
        return f"Nenhum touro encontrado na empresa {target_company}."
    
    print(f"Found {len(company_bulls)} bulls in company {target_company}")  # Debug print

    # Find the most similar bull
    min_distance = float('inf')
    similar_bull = None
    similar_bull_values = None

    for index, row in company_bulls.iterrows():
        row_pta_values = row[traits].apply(pd.to_numeric, errors='coerce').fillna(0)
        distance = euclidean(specified_pta_values, row_pta_values)
        print(f"Comparing {bull_name} with {row['ShortName']}, distance: {distance}")  # Debug print
        if distance < min_distance:
            min_distance = distance
            similar_bull = row['ShortName']
            similar_bull_values = row[traits]

    if similar_bull:
        comparison = pd.DataFrame({
            "Trait": traits,
            f"{bull_name}": specified_pta_values,
            f"{similar_bull}": similar_bull_values
        })
        comparison_table = comparison.to_string(index=False)
        return f"Touro semelhante encontrado na empresa {target_company}: {similar_bull}\n\nComparação de valores PTA:\n{comparison_table}"
    else:
        return f"Nenhum touro semelhante encontrado na empresa {target_company}."

# Function to generate chat completion
def generate_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Você é um especialista em melhoramento genético animal e bovinos leiteiros. Responda sempre em {idioma}"
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

# Streamlit app
st.title('GenMate Conversational Interface')
st.write('Faça uma pergunta sobre touros e traços genéticos.')

# Initialize session state for user query
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# Function to handle form submission
def submit_form():
    user_query = st.session_state.user_query
    # List of known companies
    known_companies = touros_df['Company'].unique()

    # List of known bull names
    known_bulls = touros_df['ShortName'].unique().tolist() + touros_df['NAABCode'].unique().tolist() + touros_df['BullKey'].unique().tolist() + touros_df['LongName'].unique().tolist()

    # Convert all known bulls to strings
    known_bulls = [str(bull) for bull in known_bulls]

    # Extract company names from the query
    company_names = extract_company_names(user_query, known_companies)
    print(f"Extracted company names: {company_names}")  # Debug print

    # Extract bull names from the query
    bull_names = extract_bull_names(user_query, known_bulls)
    print(f"Extracted bull names: {bull_names}")  # Debug print

    # Extract relevant traits from the query
    relevant_traits = extract_relevant_traits(user_query)
    print(f"Extracted relevant traits: {relevant_traits}")  # Debug print

    # Ensure only one bull per name
    if bull_names:
        bull_names = list(dict.fromkeys(bull_names))

    # Check for similarity query
    if "semelhante" in user_query.lower() or "parecido" in user_query.lower():
        # Extract specified bull name and target company
        specified_bull = None
        target_company = None
        for bull in known_bulls:
            if bull.lower() in user_query.lower():
                specified_bull = bull
                break
        for company in known_companies:
            if company.lower() in user_query.lower():
                target_company = company
                break

        if specified_bull and target_company:
            # Use relevant_traits as default if no specific traits are mentioned
            if not relevant_traits:
                relevant_traits = ["TPI", "Milk", "Fat", "FatPercent", "Protein", "ProteinPercent", "ProductionEfficiency", "ProductiveLife", "SomaticCellScore", "DaughterPregnancyRate", "Health", "PredictedTransmittingAbilityType", "UdderComposite", "FeetAndLegComposite"]
            similar_bull_response = find_similar_bull(specified_bull, target_company, relevant_traits, touros_df)
            st.session_state.response = similar_bull_response
        else:
            st.session_state.response = "Por favor, especifique o nome do touro e a empresa alvo."

        # Clear the user query
        st.session_state.user_query = ""
        return

    # Check if the user is asking for a trait definition
    if "definir" in user_query.lower() or "definição" in user_query.lower():
        trait_definitions = find_trait_definition(relevant_traits, traits_df)
        trait_definitions_combined = "\n".join(trait_definitions)
        st.session_state.response = f"Informações adicionais sobre o(s) traço(s) relevante(s):\n{trait_definitions_combined}"
        st.session_state.user_query = ""
        return

    # Filter bulls by companies and specific bull names
    if company_names:
        filtered_bulls_df = touros_df[touros_df['Company'].str.contains('|'.join(company_names), case=False, na=False)]
    else:
        filtered_bulls_df = touros_df

    if bull_names:
        filtered_bulls_df = filtered_bulls_df[
            (filtered_bulls_df['ShortName'].str.lower().isin([name.lower() for name in bull_names])) |
            (filtered_bulls_df['NAABCode'].str.lower().isin([name.lower() for name in bull_names])) |
            (filtered_bulls_df['BullKey'].str.lower().isin([name.lower() for name in bull_names])) |
            (filtered_bulls_df['LongName'].str.lower().isin([name.lower() for name in bull_names]))
        ]

    # Remove duplicates by ShortName to ensure only one bull per name
    filtered_bulls_df = filtered_bulls_df.drop_duplicates(subset=['ShortName'])

    # Extract PTA data for the filtered bulls
    pta_data = get_bull_pta(filtered_bulls_df, relevant_traits)

    # Find trait definitions
    trait_definitions = find_trait_definition(relevant_traits, traits_df)

    # Create the prompt for the language model
    pta_info_list = []
    for bull_name, pta_values in pta_data.items():
        if pta_values:
            pta_info = "\n".join([f"O valor para {trait_mapping.get(trait, trait)} para o touro {bull_name} é {value}." for trait, value in pta_values.items()])
        else:
            pta_info = f"No data found for bull {bull_name}."
        pta_info_list.append(pta_info)

    pta_info_combined = "\n\n".join(pta_info_list)
    trait_definitions_combined = "\n".join(trait_definitions)
    prompt = f"""{user_query}

    Aqui temos os valores do banco de dados:
    {pta_info_combined}

    Informações adicionais sobre o(s) traço(s) relevante(s):
    {trait_definitions_combined}
    """

    print(prompt)

    # Generate the response
    response = generate_response(prompt)

    # Display the result
    st.session_state.response = response

    # Clear the user query
    st.session_state.user_query = ""

# Form for user input
with st.form(key='query_form', clear_on_submit=True):
    st.text_input('Escreva sua pergunta aqui', key='user_query')
    submit_button = st.form_submit_button(label='Enviar', on_click=submit_form)

# Display the result below the text input
if 'response' in st.session_state:
    st.write(st.session_state.response)
