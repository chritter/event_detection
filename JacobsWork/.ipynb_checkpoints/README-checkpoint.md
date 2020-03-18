* EventDetails: Experiment with techniques to extract event details using jacobs_corpus.csv. Use nlt.ner tagging. Use networkx to connect entities and plot graph

* I. Preprocessing: Reads jacobs paper data (osfstorage-archive/replicationdata/experiment_data.json) and creates jacobs_corpus.csv.
* II. EDA: does EDA on jacobs_corpus.cvs. Analysis steps documented in notebook.s. Shows that all sentence have less than ~60 tokens. Could use NN model.

* III. SVM_Sentencetraining: SVM training as in Jacobs, with hyperparameter search. Hyperparam search took 3hrs.  But no evaluation on test set based on the parameter search.
* III. SVM_Sentencetraining2: As SVM_Sentencetraining but with small improvement but no hyperparameter tuning.
* III. SVM_Sentencetraining2: Adapted to work with latest version of spacy.
* III. SVM_Sentencetraining3: Using Spacy for POS and NER extraction instead of nltk. (version 2020) No hyperparameter tuning as too expensive for now.
*  III. SVM_Sentencetraining3: focus on M&A event only to save compute time and do hyperparameter tuning.

* SVM_Documenttraining: SVM classification of jacobs_corpus_body_labeled. 

