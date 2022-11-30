# Treball de Fi de Grau (TFG)
---
## Actualització de la setmana:
### Prediccions
Vaig executar el model amb el dataset TextVQA per veure si funcionava bé. [Aquestes](https://github.com/arcb01/GED-TFG/blob/main/outputs/textvqa_run_test_2022-11-29T22:04:30.json) van ser les prediccions. Sembla que no dona cap error i les respostes semblen coherents.

## Dataset
Vaig vulguer seguir [el tutorial de mmf per afegir un dataset](https://mmf.sh/docs/tutorials/dataset) però a dalt de tot diu que està **outdated**. Per tant, vaig decidir no mirar-me'l.  
Vaig optar per mirar com era el fitxer de configuració del dataset del TextVQA que era amb el que funcionaven les predcicons. Vaig intenetar adaptar [aquest fitxer](https://github.com/facebookresearch/mmf/blob/main/projects/m4c/configs/textvqa/defaults.yaml) perquè funcionés amb VizWiz però no va funcionar.

## Observacions
- He vist que existeix [aquest fitxer](https://github.com/facebookresearch/mmf/blob/main/projects/others/mmf_bert/configs/vizwiz/defaults.yaml) de configuració de VizWiz però per un altre model. Potser es pot mirar d'adaptar-ho?
- Quan vaig fer proves vaig veure que fallava en el que anomena "answer_processor" i mirant el codi vaig observar com si hi hagués un *processor* definit per cada tipus de dataset [(codi](https://github.com/facebookresearch/mmf/blob/main/mmf/datasets/processors/processors.py#L1434). Justament a la documentació, hi ha un [tutorial de com afegir un processor](https://mmf.sh/docs/tutorials/processors), potser s'ha d'afegir un customitzat que procesi les dades de VizWiz?
- En [aquest fitxer](https://github.com/facebookresearch/mmf/blob/main/mmf/configs/datasets/vizwiz/defaults.yaml) estan els features de vizwiz pero son d'una versió antiga de 2019. La idea seria que primer conseguís que funcionés amb aquest i ja després mira a veure si es podria actualitzat a la ultima versió de 2021.

## Feina futura
He conseguit obrir un VSCode dins del container de Docker. Llavrs, tenia pensat intentar debugar i entendre el codi i veure si d'aquesta manera puc adaptar el dataset de VizWiz. Tampoc se quan de temps hem portaria això. 


### Resources
---
[**MMF doc**](https://mmf.sh/docs/projects/m4c/)

[**MMF repo**](https://github.com/facebookresearch/mmf)
