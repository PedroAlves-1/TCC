# =============================================================================
# GERADOR DE WORD CLOUD PARA DADOS DO SCOPUS
# =============================================================================
# Este script lê um arquivo CSV exportado do Scopus, processa as palavras-chave
# e gera uma nuvem de palavras (word cloud) e um histograma de frequências.
#
# COMO USAR:
#   1. Coloque este script na mesma pasta que o seu arquivo CSV do Scopus
#   2. Ajuste os parâmetros na seção "CONFIGURAÇÕES" abaixo
#   3. Execute no terminal: python scopus_wordcloud.py
#
# DEPENDÊNCIAS (instale com pip antes de rodar):
#   pip install pandas nltk wordcloud matplotlib
# =============================================================================


# =============================================================================
# SEÇÃO 1: IMPORTAÇÕES DE BIBLIOTECAS
# =============================================================================
# Aqui carregamos todas as ferramentas que o script vai usar.
# Não é necessário alterar esta seção.

import pandas as pd                         # Leitura e manipulação de tabelas (CSV)
import matplotlib.pyplot as plt             # Geração de gráficos e histogramas
from wordcloud import WordCloud             # Criação da nuvem de palavras
from nltk.corpus import stopwords           # Lista de palavras irrelevantes (stop-words)
from nltk.stem import WordNetLemmatizer    # Reduz palavras à sua forma raiz (lemmatization)
from nltk.tokenize import word_tokenize    # Divide texto em palavras individuais (tokens)
from collections import Counter             # Conta a frequência de cada elemento
import nltk                                 # Biblioteca principal de linguagem natural
import re                                   # Expressões regulares para busca em texto
import sys                                  # Usado para encerrar o programa em caso de erro


# =============================================================================
# SEÇÃO 2: CONFIGURAÇÕES PRINCIPAIS
# =============================================================================

# --- Arquivo de entrada ---
# Nome (ou caminho completo) do arquivo CSV exportado do Scopus.
# Exemplo: "scopus_results.csv" ou "C:/Dados/minha_pesquisa.csv"
ARQUIVO_CSV = "meus_artigos_dlp.csv"

# --- Coluna a ser processada ---
# Escolha qual coluna de palavras-chave usar:
#   "Author Keywords"  → palavras definidas pelos próprios autores dos artigos
#   "Index Keywords"   → palavras indexadas automaticamente pelo Scopus
COLUNA = "Author Keywords"

# --- Ativar lemmatization? ---
# Se True: palavras similares são agrupadas na forma raiz.
#   Exemplo: "nanoparticles", "nanoparticle" → "nanoparticle"
# Se False: cada variação conta separadamente.
USAR_LEMMATIZATION = True

# --- Frequência mínima ---
# Palavras que aparecem menos vezes que este valor serão ignoradas na word cloud.
# Aumente para mostrar apenas termos mais relevantes. Valor mínimo recomendado: 2
FREQUENCIA_MINIMA = 10

# --- Número de palavras no histograma ---
# Quantas palavras (as mais frequentes) serão exibidas no gráfico de barras.
TOP_N_HISTOGRAMA = 30

# --- Dimensões da imagem da word cloud (em pixels) ---
LARGURA_IMAGEM = 1600   # Largura em pixels
ALTURA_IMAGEM  = 900    # Altura em pixels

# --- Tamanho das palavras na word cloud ---
# O tamanho de cada palavra é proporcional à sua frequência, dentro deste intervalo.
TAMANHO_MINIMO_FONTE = 10   # Tamanho mínimo (palavras menos frequentes)
TAMANHO_MAXIMO_FONTE = 180  # Tamanho máximo (palavras mais frequentes)

# --- Paleta de cores da word cloud ---
# Opções populares: "viridis", "plasma", "Blues", "Reds", "Dark2", "Set1",
#                   "tab10", "coolwarm", "magma", "cubehelix", "Spectral"
PALETA_CORES = "Dark2"

# --- Cor de fundo da word cloud ---
# Use "white", "black" ou qualquer cor CSS válida (ex: "#1a1a2e")
COR_FUNDO = "white"

# --- Nomes dos arquivos de saída ---
# Os resultados serão salvos com estes nomes na mesma pasta do script.
NOME_ARQUIVO_WORDCLOUD  = "wordcloud_output.png"
NOME_ARQUIVO_HISTOGRAMA = "histograma_output.png"
NOME_ARQUIVO_FREQUENCIAS = "frequencias.txt"


# =============================================================================
# SEÇÃO 3: STOP-WORDS CUSTOMIZADAS
# =============================================================================
# Stop-words são palavras genéricas que não agregam significado à análise
# (ex: "the", "and", "of"). Além das stop-words padrão do inglês (carregadas
# automaticamente via NLTK), você pode adicionar seus próprios termos aqui.
#
# Adicione palavras entre aspas, separadas por vírgulas. Exemplo:
#   STOPWORDS_CUSTOMIZADAS = {"study", "analysis", "review", "paper", "result"}

STOPWORDS_CUSTOMIZADAS = {
    "study", "analysis", "review", "paper", "result",
    "based", "using", "used", "effect", "effects",
    "method", "approach", "new", "high", "low","sub",
}


# =============================================================================
# SEÇÃO 4: TERMOS COMPOSTOS PROTEGIDOS
# =============================================================================
# Termos com mais de uma palavra que NÃO devem ser separados durante o
# processamento. Eles aparecerão como uma unidade na word cloud.
#
# IMPORTANTE: escreva os termos em LETRAS MINÚSCULAS.
# O script substituirá espaços por underscores internamente (ex: "fused_deposition_modeling")
# e depois reverterá para exibição correta.
#
# Exemplo:
#   TERMOS_COMPOSTOS = [
#       "digital light processing",
#       "selective laser sintering",
#       "fused deposition modeling",
#   ]

TERMOS_COMPOSTOS = [
    "digital light processing",
    "selective laser sintering",
    "fused deposition modeling",
    "additive manufacturing",
    "machine learning",
    "deep learning",
    "finite element",
    "finite element analysis",
    "molecular dynamics",
    "scanning electron microscopy",
    "transmission electron microscopy",
    "x-ray diffraction",
    "mechanical properties",
    "thermal conductivity",
]


# =============================================================================
# SEÇÃO 5: DICIONÁRIO DE SINÔNIMOS
# =============================================================================
# Use este dicionário para padronizar termos equivalentes.
# O formato é: "termo_encontrado_no_texto": "termo_substituído"
#
# Todos os termos devem estar em LETRAS MINÚSCULAS.
# Útil para unificar siglas, traduções, grafias alternativas, etc.
#
# Exemplos de uso:
#   - Siglas: "al2o3" → "alumina"
#   - Traduções: "óxido de alumínio" → "alumina"
#   - Grafias: "nano-particle" → "nanoparticle"

SINONIMOS = {
    "al2o3":              "alumina",
    "óxido de alumínio":  "alumina",
    "tio2":               "titanium dioxide",
    "zro2":               "zirconia",
    "zro":               "zirconia",
    "sio2":               "silica",
    "fe3o4":              "iron oxide",
    "cnt":                "carbon nanotube",
    "cnts":               "carbon nanotube",
    "go":                 "graphene oxide",
    "fdm":                "fused deposition modeling",
    "sls":                "selective laser sintering",
    "dlp":                "digital light processing",
    "am":                 "additive manufacturing",
    "ml":                 "machine learning",
    "dl":                 "deep learning",
    "ann":                "artificial neural network",
    "sem":                "scanning electron microscopy",
    "tem":                "transmission electron microscopy",
    "xrd":                "x-ray diffraction",
}


# =============================================================================
# SEÇÃO 6: DOWNLOAD DOS RECURSOS DO NLTK
# =============================================================================
# O NLTK precisa baixar alguns arquivos de dados na primeira execução.
# Esta seção faz isso automaticamente e silenciosamente.
# Não é necessário alterar nada aqui.

print("⏳ Verificando recursos do NLTK...")
for recurso in ["stopwords", "punkt", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(recurso, quiet=True)
print("✅ Recursos do NLTK verificados.\n")


# =============================================================================
# SEÇÃO 7: LEITURA DO ARQUIVO CSV
# =============================================================================
# O Scopus exporta CSVs com codificação UTF-8 (com BOM) ou latin-1.
# Tentamos ambas automaticamente para evitar erros de caracteres especiais.

print(f"📂 Lendo o arquivo: {ARQUIVO_CSV}")
try:
    # Primeira tentativa: UTF-8 com BOM (padrão mais comum do Scopus)
    df = pd.read_csv(ARQUIVO_CSV, encoding="utf-8-sig")
except UnicodeDecodeError:
    try:
        # Segunda tentativa: latin-1 (para arquivos mais antigos)
        df = pd.read_csv(ARQUIVO_CSV, encoding="latin-1")
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo CSV: {e}")
        sys.exit(1)

# Verifica se a coluna escolhida existe no arquivo
if COLUNA not in df.columns:
    print(f"❌ Coluna '{COLUNA}' não encontrada no CSV.")
    print(f"   Colunas disponíveis: {list(df.columns)}")
    sys.exit(1)

# Conta quantos artigos têm dados na coluna escolhida
total_artigos   = len(df)
artigos_com_kw  = df[COLUNA].notna().sum()
print(f"   ✅ {total_artigos} artigos encontrados, {artigos_com_kw} com dados em '{COLUNA}'.\n")


# =============================================================================
# SEÇÃO 8: EXTRAÇÃO DAS PALAVRAS-CHAVE
# =============================================================================
# No CSV do Scopus, múltiplas keywords de um mesmo artigo ficam separadas
# por ponto-e-vírgula (;). Esta seção separa e limpa todas elas.

print("🔍 Extraindo palavras-chave...")

# Descarta linhas sem dados e une tudo em uma lista de strings
lista_bruta = df[COLUNA].dropna().tolist()

# Quebra cada célula pelo separador ";" e limpa espaços em branco
todas_keywords = []
for celula in lista_bruta:
    termos = [t.strip().lower() for t in str(celula).split(";") if t.strip()]
    todas_keywords.extend(termos)

print(f"   ✅ {len(todas_keywords)} ocorrências de keywords extraídas.\n")


# =============================================================================
# SEÇÃO 9: SUBSTITUIÇÃO DE SINÔNIMOS
# =============================================================================
# Aplica o dicionário SINONIMOS definido na Seção 5.
# A substituição ocorre antes da tokenização para garantir que siglas e
# expressões compostas sejam convertidas corretamente.

print("🔄 Aplicando substituições de sinônimos...")
contagem_substituicoes = 0

keywords_normalizadas = []
for kw in todas_keywords:
    if kw in SINONIMOS:
        keywords_normalizadas.append(SINONIMOS[kw])
        contagem_substituicoes += 1
    else:
        keywords_normalizadas.append(kw)

print(f"   ✅ {contagem_substituicoes} substituições de sinônimos realizadas.\n")


# =============================================================================
# SEÇÃO 10: PROTEÇÃO DE TERMOS COMPOSTOS
# =============================================================================
# Termos compostos (ex: "machine learning") seriam quebrados em tokens separados
# ("machine" e "learning") durante a tokenização. Para evitar isso, substituímos
# temporariamente os espaços por underscores (ex: "machine_learning").
# Ao final, revertemos essa substituição para exibição correta.

print("🛡️  Protegendo termos compostos...")

# Ordena do maior para o menor (garante que "finite element analysis"
# seja substituído antes de "finite element")
termos_ordenados = sorted(TERMOS_COMPOSTOS, key=len, reverse=True)

# Dicionário para reverter os underscores no final
mapa_reverso = {}

keywords_protegidas = []
for kw in keywords_normalizadas:
    for termo in termos_ordenados:
        if termo in kw:
            # Cria uma versão com underscore para proteger o termo composto
            termo_protegido = termo.replace(" ", "_")
            mapa_reverso[termo_protegido] = termo  # Guarda o mapeamento inverso
            kw = kw.replace(termo, termo_protegido)
    keywords_protegidas.append(kw)

print(f"   ✅ {len(mapa_reverso)} padrões de termos compostos protegidos.\n")


# =============================================================================
# SEÇÃO 11: TOKENIZAÇÃO E REMOÇÃO DE STOP-WORDS
# =============================================================================
# "Tokenizar" significa dividir o texto em unidades individuais (palavras).
# Em seguida, removemos as stop-words (palavras sem valor semântico).

print("✂️  Tokenizando e removendo stop-words...")

# Carrega stop-words padrão do inglês
stop_en   = set(stopwords.words("english"))
# Une com as stop-words customizadas da Seção 3
todas_stops = stop_en | STOPWORDS_CUSTOMIZADAS

tokens_filtrados = []
for kw in keywords_protegidas:
    # word_tokenize separa o texto em palavras, pontuações etc.
    tokens = word_tokenize(kw)
    for token in tokens:
        token_lower = token.lower()
        # Mantém apenas palavras reais (sem pontuação) que não são stop-words
        # e que têm pelo menos 2 caracteres
        if (token_lower.isalpha() or "_" in token_lower) \
           and token_lower not in todas_stops \
           and len(token_lower) >= 2:
            tokens_filtrados.append(token_lower)

print(f"   ✅ {len(tokens_filtrados)} tokens após filtragem.\n")


# =============================================================================
# SEÇÃO 12: LEMMATIZATION (OPCIONAL)
# =============================================================================
# Lemmatization reduz palavras à sua forma canônica (lema).
# Exemplo: "nanoparticles" → "nanoparticle" | "properties" → "property"
#
# Atenção: termos compostos com underscore são preservados intactos.
# Esta etapa é controlada pelo parâmetro USAR_LEMMATIZATION (Seção 2).

if USAR_LEMMATIZATION:
    print("📚 Aplicando lemmatization...")
    lem = WordNetLemmatizer()
    tokens_lematizados = []
    for token in tokens_filtrados:
        if "_" in token:
            # Não aplica lemmatization em termos compostos protegidos
            tokens_lematizados.append(token)
        else:
            # Aplica lemmatization como substantivo (pos="n") por padrão
            # Outras opções: "v" (verbo), "a" (adjetivo), "r" (advérbio)
            tokens_lematizados.append(lem.lemmatize(token, pos="n"))
    tokens_filtrados = tokens_lematizados
    print("   ✅ Lemmatization concluída.\n")
else:
    print("ℹ️  Lemmatization desativada (USAR_LEMMATIZATION = False).\n")


# =============================================================================
# SEÇÃO 13: CONTAGEM DE FREQUÊNCIAS
# =============================================================================
# Conta quantas vezes cada palavra aparece e filtra pela frequência mínima.

print("📊 Contando frequências...")
contagem_total = Counter(tokens_filtrados)

# Filtra apenas palavras que atingem a frequência mínima configurada
contagem_filtrada = {
    palavra: freq
    for palavra, freq in contagem_total.items()
    if freq >= FREQUENCIA_MINIMA
}

# Reverte os underscores para espaços (para exibição correta)
contagem_final = {}
for palavra, freq in contagem_filtrada.items():
    # Verifica se é um termo composto protegido e reverte
    if palavra in mapa_reverso:
        palavra_display = mapa_reverso[palavra]
    else:
        # Substitui underscores restantes por espaços (caso haja)
        palavra_display = palavra.replace("_", " ")
    contagem_final[palavra_display] = freq

print(f"   ✅ {len(contagem_final)} termos únicos (freq. ≥ {FREQUENCIA_MINIMA}).\n")


# =============================================================================
# SEÇÃO 14: EXPORTAÇÃO DA LISTA DE FREQUÊNCIAS
# =============================================================================
# Salva um arquivo de texto com todas as palavras e suas frequências,
# ordenadas da mais frequente para a menos frequente.

print(f"💾 Salvando lista de frequências em '{NOME_ARQUIVO_FREQUENCIAS}'...")
palavras_ordenadas = sorted(contagem_final.items(), key=lambda x: x[1], reverse=True)

with open(NOME_ARQUIVO_FREQUENCIAS, "w", encoding="utf-8") as f:
    f.write(f"Arquivo: {ARQUIVO_CSV}\n")
    f.write(f"Coluna:  {COLUNA}\n")
    f.write(f"Total de termos únicos: {len(palavras_ordenadas)}\n")
    f.write("=" * 40 + "\n")
    f.write(f"{'Palavra':<40} {'Frequência':>10}\n")
    f.write("-" * 52 + "\n")
    for palavra, freq in palavras_ordenadas:
        f.write(f"{palavra:<40} {freq:>10}\n")

print(f"   ✅ Lista salva.\n")


# =============================================================================
# SEÇÃO 15: GERAÇÃO DO HISTOGRAMA
# =============================================================================
# Cria um gráfico de barras horizontal com os TOP_N_HISTOGRAMA termos
# mais frequentes. Útil para análise quantitativa rápida.

print(f"📈 Gerando histograma (top {TOP_N_HISTOGRAMA} palavras)...")

# Seleciona as N palavras mais frequentes para o gráfico
top_palavras = palavras_ordenadas[:TOP_N_HISTOGRAMA]
labels = [p[0] for p in top_palavras]   # Nomes das palavras (eixo Y)
valores = [p[1] for p in top_palavras]  # Frequências (eixo X)

# Configuração do tamanho da figura
# Ajuste os números se o gráfico ficar muito pequeno ou grande
fig, ax = plt.subplots(figsize=(12, max(6, TOP_N_HISTOGRAMA * 0.35)))

# Barras horizontais: as palavras mais frequentes ficam no topo
barras = ax.barh(range(len(labels)), valores, color="steelblue", edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()  # Coloca a palavra mais frequente no topo

# Adiciona o valor numérico ao final de cada barra
for barra, valor in zip(barras, valores):
    ax.text(
        barra.get_width() + 0.3,  # Posição X (logo após a barra)
        barra.get_y() + barra.get_height() / 2,  # Posição Y (centro da barra)
        str(valor),
        va="center", ha="left", fontsize=9
    )

# Títulos e rótulos do gráfico
ax.set_title(
    f"Top {TOP_N_HISTOGRAMA} Palavras-Chave Mais Frequentes",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Frequência", fontsize=11)
ax.set_ylabel("Termos", fontsize=11)

# Remove a borda superior e direita (visual mais limpo)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(NOME_ARQUIVO_HISTOGRAMA, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Histograma salvo em '{NOME_ARQUIVO_HISTOGRAMA}'.\n")


# =============================================================================
# SEÇÃO 16: GERAÇÃO DA WORD CLOUD
# =============================================================================
# Cria a nuvem de palavras usando os parâmetros definidos na Seção 2.
# O tamanho de cada palavra é proporcional à sua frequência.

print("☁️  Gerando word cloud...")

# Verifica se há palavras suficientes para gerar a cloud
if not contagem_final:
    print("❌ Nenhuma palavra encontrada com a frequência mínima configurada.")
    print(f"   Tente reduzir FREQUENCIA_MINIMA (atualmente: {FREQUENCIA_MINIMA})")
    sys.exit(1)

# Cria o objeto WordCloud com os parâmetros configurados
wc = WordCloud(
    width            = LARGURA_IMAGEM,       # Largura da imagem em pixels
    height           = ALTURA_IMAGEM,        # Altura da imagem em pixels
    background_color = COR_FUNDO,            # Cor do fundo
    colormap         = PALETA_CORES,         # Paleta de cores para as palavras
    min_font_size    = TAMANHO_MINIMO_FONTE, # Tamanho mínimo da fonte
    max_font_size    = TAMANHO_MAXIMO_FONTE, # Tamanho máximo da fonte
    min_word_length  = 2,                    # Ignora palavras com menos de 2 letras
    collocations     = False,                # Desativa bigrams automáticos (já tratamos manualmente)
    prefer_horizontal= 1,                 # 100% das palavras ficam na horizontal
    random_state     = 42,                   # Semente para reprodutibilidade do layout
).generate_from_frequencies(contagem_final)  # Usa as frequências calculadas

# Salva a word cloud como PNG
fig_wc, ax_wc = plt.subplots(figsize=(LARGURA_IMAGEM / 100, ALTURA_IMAGEM / 100))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")  # Remove os eixos (não fazem sentido em uma word cloud)

plt.tight_layout(pad=0)
plt.savefig(NOME_ARQUIVO_WORDCLOUD, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Word cloud salva em '{NOME_ARQUIVO_WORDCLOUD}'.\n")


# =============================================================================
# SEÇÃO 17: RESUMO FINAL
# =============================================================================
# Exibe um resumo dos resultados gerados.

print("=" * 55)
print("🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 55)
print(f"  📄 Arquivo processado : {ARQUIVO_CSV}")
print(f"  📌 Coluna analisada   : {COLUNA}")
print(f"  🔤 Termos únicos      : {len(contagem_final)}")
print(f"  🏆 Termo mais comum   : {palavras_ordenadas[0][0]} ({palavras_ordenadas[0][1]}x)")
print(f"\n  Arquivos gerados:")
print(f"    📊 Histograma  → {NOME_ARQUIVO_HISTOGRAMA}")
print(f"    ☁️  Word cloud  → {NOME_ARQUIVO_WORDCLOUD}")
print(f"    📝 Frequências → {NOME_ARQUIVO_FREQUENCIAS}")
print("=" * 55)
