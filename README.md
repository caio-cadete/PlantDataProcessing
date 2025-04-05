# 🌱 Plant Data Processing Pipeline

Este repositório contém um pipeline de pré-processamento de dados utilizado em um projeto de ciência de dados para **previsão de ocorrência de espécies vegetais** no estado do **Rio de Janeiro**. Ele envolve o cruzamento de informações de espécies vegetais com dados meteorológicos, climáticos e edáficos (solo), estruturando o dataset final para o treinamento de modelos preditivos.

---

## 🚀 Como começar

1. Clone o repositório:
```bash
git clone https://github.com/caio-cadete/PlantDataProcessing.git
cd PlantDataProcessing
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```


📚 Fontes de Dados
🌱 Dados de Plantas
Portal SiBBr (ALA-Hub Brasil)
Link: https://ala-hub.sibbr.gov.br/
Dados coletados:

Espécies do Reino Plantae

Registros geolocalizados no estado do Rio de Janeiro

Informações taxonômicas completas (ordem, família, gênero, espécie)

Data de ocorrência e localização

🌍 Dados Climáticos, de Solo e Bioma
Fonte: TRY Database – Site Climate and Soil Information
Arquivo: TRY_6_Site_Climate_Soil.zip

Clima:
Hijmans, R.J., Cameron, S.E., Parra, J.L., Jones, P.G., & Jarvis, A. (2005).
Very high resolution interpolated climate surfaces for global land areas.
International Journal of Climatology, 25: 1965–1978.

Classificação Climática:
Peel, M. C., Finlayson, B. L., & McMahon, T. A. (2007).
Updated world map of the Köppen-Geiger climate classification.
Hydrol. Earth Syst. Sci., 11, 1633–1644.

Solo:
FAO/IIASA/ISRIC/ISSCAS/JRC (2012).
Harmonized World Soil Database v1.2.
FAO, Rome, Italy and IIASA, Laxenburg, Austria.

Referência geral:
Kattge, J. et al. (2011).
TRY - a global database of plant traits.
Global Change Biology, 17: 2905–2935.

✨ Objetivo final
A finalidade deste pipeline é gerar um dataset estruturado que servirá de base para treinamento de modelos preditivos capazes de indicar quais espécies de plantas têm maior probabilidade de ocorrência em determinadas áreas do RJ, com base em clima, solo e localização geográfica.

👨‍💻 Autor
Caio Victor Soares Cadete
