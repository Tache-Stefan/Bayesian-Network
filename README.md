# Retele Bayesiene

## Introducere

### Ce sunt Rețelele Bayesiene?

O **rețea bayesiană** (Bayesian Network) este un model grafic probabilistic care reprezintă variabilele și dependențele probabilistice dintre ele prin intermediul unui graf aciclic orientat (DAG - Directed Acyclic Graph).

#### Componentele Principale:

1. **Noduri (Nodes)**: Reprezentează variabilele aleatoare
2. **Muchii Orientate (Directed Edges)**: Indică relații de dependență cauzală. O muchie de la $X$ la $Y$ înseamnă că $X$ influențează probabilitatea lui $Y$
3. **Tabele de Probabilitate Condiționată (CPT)**: Pentru fiecare nod, se calculează $P(X | \text{Parents}(X))$

## Descrierea Proiectului

### Obiectiv General

Acest proiect implementează un **pipeline automat de învățare a structurii rețelelor bayesiene** dintr-un dataset real. Folosind algoritmi avansați de învățare, proiectul extrage relații cauzale din date și construiește o rețea bayesiană completă cu:

- Structura DAG (graf aciclic orientat)
- Tabele de probabilitate condiționată (CPT)
- Reprezentare vizuală a rețelei

## Abordare

### Faza 1: Preprocesarea Datelor

**Fișier**: `src/data_processor.py`

Datele sunt preprocesate prin:
- Încărcarea și validarea datelor
- Redenumirea coloanelor cu nume mai descriptive
- **Discretizarea variabilelor continue**: Transformarea valorilor continue în categorii discreate (deoarece algoritmii folosiți lucrează cu variabile discrete)
- Normalizare și curățare

### Faza 2: Învățarea Structurii - Algoritm MMPC

**Fișier**: `src/mmpc.py`

Se utilizează algoritmul **MMPC (Max-Min Parents-Children)** pentru a construi **scheletul** rețelei (muchiile neorientate).

#### Cum Funcționează MMPC:

1. **Pentru fiecare variabilă țintă** (target):
   - Se găsesc **părinții și copiii** (Parents and Children - PC)
   - Se folosesc **teste de independență condiționată** (χ² - G-test)
   - Se selectează iterativ variabilele cu cea mai puternică asociere

2. **Testele de Independență**:
   - Testul χ² (G-square test) cu nivel de semnificație α = 0.05
   - Dacă valoarea p > α, variabilele sunt independente condiționat

3. **Output**: Un set de muchii neorientate care formează scheletul rețelei

### Faza 3: Orientarea Muchiilor - Hill Climbing Search

**Fișier**: `src/hill_climbing.py`

După obținerea scheletului, se determină **direcția muchiilor** folosind **Hill Climbing Search** cu scoring BIC.

#### Proces:

1. **Pornire**: Se pleacă cu graful gol
2. **Iterații**: 
   - Se evaluează operații posibile: **ADD** (adăugare muchie), **REMOVE** (ștergere muchie), **REVERSE** (inversare muchie)
   - Se alege operația care **maximizează** scorul BIC
   - Se aplică operația și se continuă

3. **Criterii de Validitate**:
   - Se respectă scheletul (muchiile există doar între nodurile legate în schelet)
   - Se evită ciclurile (pastrând proprietatea DAG)
   - Se limitează în-degree-ul (maxim 4 părinți per nod)

#### Scorul BIC (Bayesian Information Criterion):

$$\text{BIC} = \log P(\text{Data} | \text{Structure}) - \frac{k}{2} \log n$$

Unde:
- $\log P(\text{Data} | \text{Structure})$ = likelihood-ul datelor
- $k$ = numărul de parametri
- $n$ = numărul de observații

### Faza 4: Calculul Tabelelor de Probabilitate Condiționată

**Fișier**: `src/probability.py`

După obținerea structurii finale, se calculează **probabilitățile condiționate** pentru fiecare nod:

$$P(X | \text{Parents}(X)) = \frac{\text{count}(X, \text{Parents}(X))}{\text{count}(\text{Parents}(X))}$$

Rezultatele sunt salvate în `output/CPT.txt`.

---

## Structura Proiectului

```
Bayesian-Network/
├── main.py                          # Script principal de execuție
├── README.md                        # Documentație (acest fișier)
├── data/
│   ├── raw/
│   │   └── heart.csv               # Dataset original
│   └── processed/
│       └── heart_discrete.csv      # Date discretizate și procesate
├── src/
│   ├── data_processor.py           # Preprocesare și discretizare date
│   ├── mmpc.py                     # Algoritm MMPC pentru învățarea scheletului
│   ├── hill_climbing.py            # Hill Climbing pentru orientarea muchiilor
│   ├── graph.py                    # Structura de date DAG
│   ├── scoring.py                  # Calculul scorului BIC
│   ├── probability.py              # Calculul CPT
│   └── stats.py                    # Teste statistice 
└── output/
    ├── bayes_net.png               # Reprezentare vizuală a rețelei
    └── CPT.txt                     # Tabelele de probabilitate condiționată
```

## Instalare și Utilizare

### Cerințe

Vezi fișierul requirements.txt pentru lista completă de dependințe.

### Instalare Dependențe

Se recomandă crearea unui mediu virtual și instalarea dependențelor din requirements.txt:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# sau (Command Prompt)
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Rulare Principală

```bash
python main.py
```
