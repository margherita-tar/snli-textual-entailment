# SNLI Textual Entailment
**1. INTRODUZIONE**

---

1.1 SPIEGAZIONE DEL PROGETTO e OBIETTIVO DEL LAVORO
Il progetto affronta un classico task di Natural Language Processing (NLP) noto come Natural Language Inference (NLI). L'obiettivo è determinare la relazione logica tra una coppia di frasi: una "premessa" e un'"ipotesi".

Il modello deve classificare questa relazione in una delle seguenti tre categorie:

Entailment (Conseguenza): l'ipotesi è una diretta e logica conseguenza della premessa.
Contradiction (Contraddizione): l'ipotesi contraddice l'informazione contenuta nella premessa.
Neutral (Neutrale): l'ipotesi potrebbe essere vera, ma non è né una conseguenza diretta né una contraddizione della premessa.
Per raggiungere questo scopo, il progetto esplora e confronta due approcci di complessità crescente:

Modello Baseline: una Regressione Logistica che utilizza feature ingegnerizzate dal testo (TF-IDF e similarità coseno).
Modello Avanzato: una rete neurale ricorrente, specificamente una Bi-directional Long Short-Term Memory (Bi-LSTM), in grado di catturare il contesto e le dipendenze sequenziali all'interno delle frasi.
Il framework utilizzato è PyTorch con l'ausilio di PyTorch Lightning per semplificare e standardizzare i cicli di addestramento e valutazione.

1.2 DESCRIZIONE DEL DATASET
SNLI è un dataset di grandi dimensioni:

Set di Addestramento (Train): ~550.000 esempi
Set di Validazione (Dev): ~10.000 esempi
Set di Test: ~10.000 esempi
L’esaurimento della RAM durante l'esecuzione ci ha indotto successivamente a ridurre le dimensioni del dataset, con l’obiettivo di mantenere comunque un buon trade-off tra velocità dell’esecuzione del modello e performance.

Descrizione delle Features del Dataset SNLI:

Nome Colonna	Descrizione	Esempio
gold_label	Etichetta finale (target). Indica la relazione logica tra sentence1 e sentence2.	entailment, contradiction, o neutral
sentence1	Frase di premessa.	A person on a horse jumps over a broken down airplane.
sentence2	Frase di ipotesi. Generata da un annotatore per confrontarsi con la premessa.	A person is training his horse for a competition.
sentence1_parse	Analisi sintattica della premessa.	(ROOT (S (NP (DT A) (NN person)) ...
sentence2_parse	Analisi sintattica dell'ipotesi.	(ROOT (S (NP (DT A) (NN person)) ...
sentence1_binary_parse	Parse binarizzata della premessa.	( ( ( A person ) ( on ( a horse ) ) ) ...
sentence2_binary_parse	Parse binarizzata dell'ipotesi.	( ( A person ) ( is ( training ( his horse ) ) ) )
captionID	ID della didascalia.	3416050480.jpg#4
pairID	ID univoco per la coppia premessa-ipotesi.	3416050480.jpg#4r1n
label1 ... label5	Etichette date da 5 annotatori.	La maggior parte sono NaN
**2. SET-UP**
installazione dell’ambiente pytorch lightening
import delle librerie
download del dataset in formato zip, estrazione e successiva visualizzazione
**3. ANALISI DEI DATI (EDA & PLOTS)**
EDA:

d.shape: stampa delle dimensioni dei set dei dati e le relative proporzioni in percentuale

d.types: le features sono tutte object, ovvero non numeriche. Nel nostro caso specifico sono stringhe

distribuzione delle classi nella gold label:

Osservazioni:

le 3 classi di interesse sono bilanciate
presenza di una classe di label anomala rappresentata da “-”, ad indicare che c’è disaccordo tra annotatori (label1, label2 ecc) sull’etichetta corretta da assegnare per una determinata coppia di frasi. Procederemo successivamente con il filtraggio di questo valore prima dell’utilizzo del dataset per addestramento
creazione di una funzione da chiamare per ognuno dei tre sub-set:

per semplificare il dataset e centrare meglio il nostro obiettivo, abbiamo deciso di eliminare sia le features inutili nella previsione (captionID e pairID), sia quelle altamente specifiche che avrebbero richiesto l’utilizzo di modelli di complessità avanzata: le variabili “parse” rappresentano la struttura sintattica della frase, mentre le variabili label costituiscono l’insieme delle etichette date da 5 annotatori diversi.
rimozione NaN per le feautures
filtraggio delle label valide
mapping delle gold label in valori numerici
stampa della lunghezza media delle frasi per ogni dataset per valutare eventuali sbilanciamenti che non sono emersi
plot della distribuzione delle classi per ogni dataset per valutare eventuale sbilanciamento
**4. LOGISTIC REGRESSION (MODELLO BASE)**
Come primo modello base implementiamo una Regressione Logistica OvR, di cui abbiamo valutato le performance tramite metriche di valutazione e plot. Questo modello sarà poi confrontato con una versione avanzata basata su LSTM.

4.1 PRE PROCESSING (SAMPLING) E FEATURE ENGINEERING (VETTORIZZAZIONE)
Campionamento (Sampling): invece di utilizzare l'intero dataset, viene estratto un campione casuale del 50% da ciascun set (training, validation e test) per iterare più velocemente e per evitare problemi con la RAM
Vettorizzazione del testo: utilizzando la Tecnica TF-IDF, si convertono documenti testuali in una matrice di feature numeriche basate sulla frequenza dei termini (TF) e sulla frequenza inversa del documento (IDF): si assegna quindi un peso maggiore alle parole che sono frequenti in una frase ma rare nell'intero corpus. Il risultato sono due matrici sparse
Parametri del Vettorizzatore:
max_features=3000: limita il vocabolario alle 3000 parole più frequenti (in base al loro peso TF-IDF). Aiuta a controllare la dimensionalità e a rimuovere parole molto rare.

Nota: dal momento che si tratta del modello base, l'obiettivo è un trade-off tra velocità dell'esecuzione e buona performance di partenza. Tale parametro verrà modificato con un valore più alto nel modello implementato successivamente.

min_df=5: ignora le parole che appaiono in meno di 5 documenti, per filtrare errori di battitura o termini estremamente rari.

max_df=0.9: ignora le parole che appaiono in più del 90% dei documenti (es. stop-words molto comuni non catturate da altre liste).

Feature Aggiuntiva (Similarità Coseno): viene calcolata la similarità coseno tra i vettori TF-IDF di sentence1 e sentence2 (se due documenti hanno molte parole in comune, il valore di cosime similarity sarà alto).
4.2 HYPERPARAMETER TUNING
Attraverso GridSearchCV, si è fatta una ricerca per i migliori iperparametri. La griglia è composta dai seguenti parametri testati:
C: [0.01, 0.1, 1, 10]
penalty: ['l2']
solver: ['lbfgs', 'saga']
max iter: [500]
Il totale delle combinazioni testate è riportato nell'output, attraverso 'verbose=1': 4 (C) × 1 (penalty) × 2 (solver) = 8 combinazioni, con 5-fold CV, per un totale di 40 fits.

Mantenendo fissi i valori dei parametri scelti arbitrariamente (max iter = 500 e penalty = l2), la configurazione migliore dei restanti è la seguente:

C = 1
solver = 'saga'
4.3 REGRESSIONE LOGISTICA OvR
Il modello implementa una strategia One-vs-Rest, quindi addestra 3 classificatori binari indipendenti:
Contradiction vs not-Contradiction
Entailment vs not-Entailment
Neutral vs not-Neutral
La valutazione dettagliata del modello avviene attraverso:
Classification Report: stampato per tutti e tre i set di dati, fornisce le metriche di precision, recall, F1-score per ogni classe
Matrice di Confusione
Curva ROC e AUC: la curva ROC mostra il trade-off tra Tasso di Veri Positivi e Tasso di Falsi Positivi. L'area sotto la curva (AUC) è una misura aggregata della performance del classificatore per quella classe.
Tabella Riassuntiva - Osservazioni & Analisi del Modello:

Categoria	Osservazione
Prestazioni per Classe	Entailment ha il miglior F1-score su tutti i dataset, ad esempio su Test Set, F1-score = 0.72, mentre recall = 0.74; Neutral è la classe più debole con F1 intorno a 0.65 e recall 0.62 su Test Set.
Generalizzazione	Le metriche su train, validation e test sono coerenti, senza segni evidenti di overfitting.
Accuracy	Accuracy stabile tra 68.2% (test) e 68.3% (train); leggero calo su validation (67.6%), ma nel complesso costante.
Curva ROC (AUC)	Le curve ROC mostrano buona separazione tra classi, con AUC = 0.86 (Entailment), 0.85 (Contradiction), 0.82 (Neutral).
Conclusione:

A seguito di vari tentativi, modificando per esempio i parametri del modello (numero massimo di features, frazionamento del dataset), questa versione è sembrata un buon punto di partenza per raggiungere un trade-off discreto per un dataset di una complessità non indifferente e un modello standard, che coglie una relazione lineare semplice.

**5. LSTM PER NLI**
Come secondo modello abbiamo deciso di testare un approccio più avanzato per ottenere performance migliori tramite implementazione di una rete neurale ricorrente, specificamente una BiLSTM (Bidirectional Long Short-Term Memory), per catturare le relazioni semantiche e sequenziali tra due frasi.

5.1 PRE PROCESSING E FEATURE ENGINEERING (VETTORIZZAZIONE E COSTRUZIONE DEL VOCABOLARIO)
Mantenendo la stessa percentuale di frazionamento del dataset al 50%, abbiamo ripetuto la vettorizzazione tramite TF-IDF, questa volta incrementando il numero di features, per aumentare la complessità del modello, accettando un compromesso con la diminuzione della velocità dell'esecuzione. Abbiamo proceduto creando una mappa numerica (word2idx) che associa ogni parola a un indice intero. Questo è un passo fondamentale per poter fornire il testo a un modello neurale:

Filtri: Il vocabolario è limitato alle 10.000 parole più comuni che appaiono almeno 2 volte nel corpus di training+development. Questo riduce il rumore e la dimensionalità.
Token Speciali: Vengono aggiunti due token cruciali:
(indice 0): usato per riempire le frasi e portarle tutte alla stessa lunghezza all'interno di un batch.
(indice 1): sostituisce le parole non presenti nel vocabolario (Out-Of-Vocabulary).
Dopo di che, per ogni coppia di frasi (campione idx) si eseguono le seguenti operazioni:

Tokenizzazione e Indicizzazione: si trasformano le frasi da testo a sequenze di indici numerici usando il word2idx creato in precedenza.
Feature Engineering: calcolo di 3 feature numeriche esplicite che forniscono al modello segnali aggiuntivi:
Similarità Coseno: misura la somiglianza semantica basata su TF-IDF.
Overlap di Parole: misura quante parole sono in comune tra le due frasi.
Differenza di Lunghezza: misura la differenza nel numero di parole.
Per risolvere il problema dei batch di frasi con lunghezza diverse si implementa la funzione collate_fn, che tramite padding (aggiungendo valore 0 alla fine delle frasi più corte), rende le frasi tutte le della stessa lunghezza, pronte per essere elaborate dalla rete neurale.

5.2 ARCHITETTURA DEL MODELLO NEURALE EntailmentLSTMModel
Abbiamo scelto di usare 2 hidden layers poiché, dopo aver valutato le performance con un solo strato interno si è notato solo un lievissimo miglioramento delle metriche rispetto alla Logistic Regression.

La rete neurale è composta dai seguenti layer:

nn.Embedding: converte gli indici delle parole in vettori densi (embedding) di dimensione 200. Questi vettori catturano le relazioni semantiche tra le parole. Il padding_idx=0 assicura che i token di padding non influenzino i calcoli.
nn.LSTM: layer principale. È una BiLSTM:
Bidirezionale: elabora ogni frase in entrambe le direzioni forward e backward (rispettivamente da sinistra a destra e da destra a sinistra) per catturare il contesto completo di ogni parola.
Stacked a 2 strati: aumenta la capacità del modello di apprendere rappresentazioni più complesse e astratte
Si calcolano tre rappresentazioni per ogni frase:

L’ultimo hidden state della direzione forward
Il primo hidden state della direzione backward
Un max pooling su tutti i timestep (per catturare le feature più salienti).
Le rappresentazioni delle due frasi (h1 e h2) vengono combinate in modo sofisticato per catturare la loro interazione:

Concatenazione di h1 e h2.
La loro differenza assoluta (abs(h1 - h2)).
Il loro prodotto elemento per elemento (h1 * h2).
Questo vettore di interazione è arricchito con 3 feature manuali:

Similarità coseno tra le frasi
Overlap (percentuale di parole in comune)
Differenza di lunghezza normalizzata
Passa poi attraverso un layer di Dropout (per la regolarizzazione) che serve a disattivare casualmente alcuni neuroni durante il training, riducendo il rischio di overfitting. 3. nn.Linear: layer finale di classificazione, che mappa il vettore ad uno spazio di dimensione 3, corrispondente alle 3 classi target. Il layer produce i logit, valori grezzi non normalizzati che rappresentano la propensione del modello verso ciascuna classe. Si applicherà la funzione softmax per ottenere probabilità.

5.3 ADDESTRAMENTO
Come nel modello precedente, anche questo addestramento è gestito dal Trainer che automatizza l’intero processo, munito anche di nuove soluzioni:

EarlyStopping: è un "callback" che monitora la loss di validazione (val_loss). Se la loss non migliora per 3 epoche consecutive (patience=3), l'addestramento viene interrotto automaticamente. Questo previene l'overfitting e fa risparmiare tempo.
LossCallback: un callback personalizzato che salva i valori di loss di training e validazione alla fine di ogni epoca. Questi dati vengono poi utilizzati per plottare le curve di apprendimento.
AdamW: un’evoluzione dell'ottimizzatore Adam che spesso porta a una migliore generalizzazione. Usa un learning rate preso dagli iperparametri e applica un weight decay (penalizzazione L2).
5.4 VALUTAZIONE E ANALISI DEI RISULTATI
Abbiamo deciso di valutare il modello sulle stesse metriche usate per la Logistic Regression così da rendere più agevole il confronto tra le due.

Tabella Riassuntiva - Osservazioni & Analisi Modello

Categoria	Osservazione
Prestazioni per Classe	Entailment ha il miglior F1-score su tutti i dataset mantenendo il trend descritto dai risultati di Logistic Regression; Neutral rimane la classe più debole.
Generalizzazione	Le metriche su train, validation e test mantengono la coerenza, senza segni evidenti di overfitting.
Accuracy	L'accuracy di Test e Validation Set si aggirano intorno a 72%, mentre non sorprende che sia leggermente più alta sul Train Set (~75,5%)
Curva ROC (AUC)	Le curve ROC mostrano buona separazione tra classi, con AUC = 0.90 (Entailment), 0.88 (Contradiction), 0.86 (Neutral).
Conclusione:

A differenza della Regressione Logistica semplice, con LSTM abbiamo avuto l'opportunità di concentrarci su nuove tecniche di ottimizzazione: per esempio, abbiamo sperimentato manualmente valori differenti di weight_decay per valutare l'effetto della regolarizzazione L2 sulle capacità di generalizzazione del modello, con l'obiettivo di ridurre l'overfitting osservato dopo alcune epoche di training:

Sono stati testati valori come 0, 1e-5, 1e-4, 1e-3 e 1e-2.
AdamW integra nativamente il weight decay, migliorando la stabilità della regolarizzazione rispetto all'uso classico dell’ottimizzatore Adam con penalizzazione L2.
In aggiunta, per valutare l'andamento dell'apprendimento e quindi la diminuzione delle loss, si è proceduto con il plot delle Learning Curves, a riprova anche del mancato overfitting. Si nota un andamento di discesa delle loss fino alla terza epoca, dopo di che, la validation loss procede con una capacità di apprendimento lievemente più bassa rispetto a quella di train, continuando a scendere fino quasi ad appiattirsi. Tuttavia, come già detto, non si verifica overfitting o, per lo meno, il gap che si crea tra le due curve non è un segnale preoccupante.

**6. CONFRONTO PRESTAZIONI - MODELLO BASE vs LSTM**
Dataset	Modello Base	LSTM
Train Accuracy	0.6826	0.7552
Val Accuracy	0.6755	0.7246
Test Accuracy	0.6824	0.7215
Area Under Curve (AUC - ROC) per Classe - Set di Test:

Classe	Base	LSTM
Contradiction	0.85	0.88
Entailment	0.86	0.90
Neutral	0.82	0.86
Il modello LSTM supera nettamente il modello base su tutti i set (train, val, test), in termini di accuratezza globale, indicando una migliore capacità di apprendere e generalizzare.
Tutte e tre le classi beneficiano di questo, con incrementi evidenti in precision, recall e F1-score. In particolare, la classe Entailment è quella che mostra il miglioramento più marcato nel recall e nell’AUC. Anche le prestazioni sulla classe Neutral, notoriamente la più difficile, aumentano.
Le curve ROC confermano questi risultati: l’area sotto la curva migliora per ogni classe. Questo suggerisce che l’uso dell’LSTM per il processamento sequenziale delle frasi consente al modello di catturare dipendenze temporali e relazioni semantiche più profonde, che il modello di Regressione Logistica non riesce a cogliere pienamente.
L’LSTM ha molti più parametri e quindi più capacità di modellare dati complessi, dunque nel caso del dataset SNLI, molto vasto, questa proprietà offre vantaggi non indifferenti.
**7. CONSIDERAZIONI FINALI**
Alla luce dei risultati ottenuti emerge con chiarezza che la capacità di modellare il contesto sequenziale è un requisito fondamentale per il task di Natural Language Inference.

La differenza di performance tra un modello "bag-of-words" (lavora su rappresentazioni statiche, non tiene conto dell'ordine delle parole né della struttura sintattica delle frasi) e un modello sequenziale dimostra l’importanza dell'informazione contenuta nella sintassi e nell'ordine delle parole.

Difatti NLI non è un semplice problema di "matching" di parole chiave, ma un compito di comprensione del testo a un livello più profondo.

Se volessimo migliorare le prestazioni generali, uno step successivo potrebbe essere quello di applicare Word embedding pre-addestrati aggregati (per esempio GloVe o Word2Vec) o rappresentazioni dense (per esempio BERT), in quanto catturano aspetti semantici o sintattici dell’intera frase, seppur attraverso un modello standard.

Nessuna delle due strategie usate è esente dal commettere errori in quanto un’analisi qualitativa rivelerebbe probabilmente che dove la Regressione Logistica fallisce sistematicamente su esempi che dipendono da negazioni, quantificatori ("tutti", "nessuno") e inversione di soggetto-oggetto, la Bi-LSTM commette invece errori più circoscritti a casi di ambiguità lessicale, ragionamento multi-passo o frasi che richiedono conoscenza esterna al dataset.

Nonostante LSTM migliori le prestazioni, la classe Neutral rimane la più complessa da individuare, in quanto rappresenta una relazione tra frasi sfumata e meno definita. Si potrebbe pensare a tecniche più avanzate per affrontare meglio la classificazione di questa classe (per esempio: pesi di loss, sampling).
