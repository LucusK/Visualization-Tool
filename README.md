# Embedding-Visualization-Tool

In my existing project, I use indexing and retrieval systems such as ColBERT and MuVERA to rank documents based on query similarity. These systems produce such similar scores which makes me feel these rankings are unintuitive, and it is difficult to understand why a particular document was retrieved. My work is mainly on the generation of the embeddings, then using similarity between two large vectors to determine matches. 
For this project, I want to build a visualization tool that explains retrieval decisions by showing the token level similarities. Given a query and a retrieved document, the tool will visualize which individual query tokens are more dominant, allowing me to see which tokens are used for retrieval and what information is lost. I plan to extend my existing pipeline to generate these visualizations and maybe observe patterns in natural language. 

I plan to make a similarity heatmap, where is query tokens x document tokens, with each cell being their similarity


In future, i want to build using among jdbc, sql, agile, php, python:
An app that takes your files, create embeddings and allow you to search. This search will result in the top 10 searches, and give the visualization for why they are chosen.


# To install dependencies
py -3.11 -m pip install -r requirements.txt
