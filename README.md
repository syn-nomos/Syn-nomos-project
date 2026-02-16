# Syn-nomos-project: Legal Data Processing Toolbox

Welcome to the **Syn-nomos-project** repository. This toolbox serves as a central hub for technical solutions developed to support large-scale extraction, preprocessing, and management of legal data using Machine Learning (ML) and Natural Language Processing (NLP) techniques.

The repository is organized into distinct functional modules (Toolboxes), ensuring that each tool is easily identifiable by the purpose within the processing pipeline.

## 📂 Repository Structure

The project is structured into three main toolboxes, each dedicated to specific scientific tasks:

### 1. Classification Toolbox
*Located in:* `/Classification Toolbox`
This module focuses on preparing and enriching data for **Text Classification** tasks.
*   **Key Components:**
    *   `ELD_generation`: Enriched Label Description Generator using LLMs (via OpenRouter API) to create semantic descriptions for legal concepts (EuroVoc, HellasVoc, Mimic).
    *   `create_dataset`: Scripts for transforming raw resources into standardized JSONL formats and generating train/dev/test splits.

### 2. NER Toolbox
*Located in:* `/NER Toolbox`
This module concentrates on producing high-quality annotated data for **Named Entity Recognition (NER)**.
*   **Key Components:**
    *   `Semi-automated Annotation Framework`: A Human-in-the-Loop (HITL) application built with Streamlit for supervising and correcting algorithm predictions.

### 3. General Purpose Toolbox
*Located in:* `/General Purpose Toolbox`
A designated space for cross-functional utilities and future general-purpose scripts that apply across different domains of the project.

---

## 🛠️ Technical Requirements & Installation

The toolbox is designed for portability across Windows, Linux, and macOS.

### Prerequisites
*   **Python**: Version 3.x
*   **Package Manager**: `pip`

### General Setup
To set up the environment for the general toolbox:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/syn-nomos/Syn-nomos-project.git
    cd Syn-nomos-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

> **Note:** Each toolbox folder (e.g., `Classification Toolbox`) may contain its own specific `requirements.txt` and `README.md` with detailed instructions for that specific module.

---

# Περιγραφή Έργου (Greek Description)

Η παρούσα εργαλειοθήκη, η οποία φιλοξενείται στο οικοσύστημα του GitHub, αποτελεί το κεντρικό αποθετήριο των τεχνικών λύσεων που αναπτύχθηκαν κατά τη διάρκεια και για τις ανάγκες του έργου *Syn-nomos*. Αποτελεί μια ολοκληρωμένη και τυποποιημένη εργαλειοθήκη (Toolbox), σχεδιασμένη για την εξαγωγή, προεπεξεργασία και διαχείριση νομικών δεδομένων σε μεγάλη κλίμακα.

Η αρχιτεκτονική του αποθετηρίου έχει δομηθεί με βάση τις λειτουργικές απαιτήσεις και τα διακριτά επιστημονικά καθήκοντα του έργου. Συγκεκριμένα, η οργάνωση τους στο αποθετήριο έχει να κάνει με την περιοχή εφαρμογή τους, επιτρέποντας την αυτόνομη ανάπτυξη και χρήση εργαλείων που σχετίζονται με τη διαχείριση Νομοθετημάτων με μεθόδους μηχανικής μάθησης και Επεξεργασίας Φυσικής Γλώσσας. Η προσέγγιση αυτή διασφαλίζει ότι κάθε εργαλείο είναι άμεσα αναγνωρίσιμο ως προς τον σκοπό του και την υπηρεσία που προσφέρει στο συνολικό pipeline επεξεργασίας.

Βασική προτεραιότητα κατά τον σχεδιασμό των εργαλείων υπήρξε η **επαναχρησιμοποίηση (reusability)**. Τα δεδομένα και οι πόροι έχουν υποστεί επεξεργασία και τυποποίηση με τέτοιο τρόπο, ώστε να αποτελούν «έτοιμες προς χρήση» (ready-to-use) εισόδους. Με αυτόν τον τρόπο, η εργαλειοθήκη καθίσταται προσβάσιμη και αξιοποιήσιμη από οποιονδήποτε τρίτο ερευνητή ή φορέα επιθυμεί να εκτελέσει παρόμοιες εργασίες ML/NLP στον νομικό τομέα, προσφέροντας σταθερά και αναπαραγώγιμα αποτελέσματα.

### Στο πλαίσιο αυτό, περιλαμβάνονται:
*   **Εξειδικευμένα scripts μετασχηματισμού**, που ενοποιούν διαφορετικούς νομικούς πόρους (π.χ. EuroVoc, ELI) σε κοινά πρότυπα αρχείων.
*   **Ροές εργασίας (Workflows) προετοιμασίας δεδομένων**, που υποστηρίζουν πολλαπλούς μορφότυπους (token-level και span-based), εξασφαλίζοντας συμβατότητα με σύγχρονους tokenizers και γλωσσικά μοντέλα (LLMs).
*   **Εργαλεία ποιοτικού ελέγχου και επισημείωσης**, ενσωματώνοντας διαδικασίες Human-in-the-Loop (HITL) για τη διασφάλιση της ακρίβειας των παραγόμενων δεδομένων.

Συνολικά, ο κώδικας λειτουργεί ως ένα δυναμικό πλαίσιο (framework) που περιγράφει όχι μόνο το «τι» παρήχθη, αλλά και το «πώς» οι νομικοί πόροι μετασχηματίζονται, ελέγχονται και οργανώνονται. Η διάρθρωση αυτή επιτρέπει την εύκολη επέκταση της εργαλειοθήκης με νέα δεδομένα, διατηρώντας την ιχνηλασιμότητα και την ποιότητα που απαιτείται.

## 1. Αποθετήριο GitHub & Οργάνωση Κώδικα

### 1.1 Δομή φακέλων
Το σύνολο του κώδικα, των εργαλείων και της τεκμηρίωσης της εργαλειοθήκης φιλοξενείται συγκεντρωμένα στον αποθετήριο με όνομα **Syn-nomos-project** στην πλατφόρμα GitHub και μέσα στο γενικότερο αποθετήριο που αφορά τα δημιουργηθέντα tasks του έργου (https://github.com/syn-nomos). Το αρχικό αποθετήριο αυτό λειτουργεί ως ο κεντρικός κόμβος του έργου, εξασφαλίζοντας την ενιαία διαχείριση, τη διαφάνεια και την άμεση πρόσβαση στα επιμέρους αποθετήρια που αναπτύχθηκαν για τις ανάγκες των διαφορετικών ερευνητικών εργασιών.

### 1.2 Ανάλυση Αποθετηρίων και Λειτουργικών Ενοτήτων
Το αποθετήριο Syn-nomos-project είναι οργανωμένο σε διακριτές λειτουργικές ενότητες, καθεμία από τις οποίες ανταποκρίνεται σε ένα συγκεκριμένο επιστημονικό και πρακτικό ζητούμενο του έργου.

#### 1.2.1. Ενότητα Ταξινόμησης (Classification Toolbox)
Η ενότητα αυτή περιλαμβάνει μια σειρά εργαλείων για την προετοιμασία και τον εμπλουτισμό δεδομένων που προορίζονται για εργασίες θεματικής ταξινόμησης (Text Classification). Τα κύρια εργαλεία είναι:
*   **Enriched Label Description Generator Tool**: Πρόκειται για μια ενοποιημένη εφαρμογή Python που αξιοποιεί Μεγάλα Γλωσσικά Μοντέλα (LLMs) μέσω του OpenRouter API για τη δημιουργία σημασιολογικών περιγραφών των νομικών εννοιών. Το εργαλείο υποστηρίζει πολλαπλά σύνολα δεδομένων (όπως EuroVoc, HellasVoc και Mimic), επιτρέποντας την αυτόματη ενσωμάτωση της ιεραρχικής δομής των όρων στο prompt για την παραγωγή πιο πλούσιων περιγραφών. Διακρίνεται για την ευελιξία του στην εναλλαγή μοντέλων (π.χ. GPT-4o, Mistral) και την ανθεκτικότητά του (error handling, retries), καθιστώντας τη διαδικασία δημιουργίας λεξιλογίων αξιόπιστη και κλιμακώσιμη.
*   **Dataset Creation & Transformation Scripts**: Μια συλλογή από scripts (εντοπίζονται στον φάκελο `Classification Toolbox/create_dataset`) που αναλαμβάνουν τη μετατροπή των ακατέργαστων πόρων σε τυποποιημένες μορφές εισόδου. Αυτό περιλαμβάνει τον μετασχηματισμό δεδομένων από συμπιεσμένες μορφές (zip) σε αρχεία JSONL, την οργάνωση των splits (train/dev/test) και την εκτέλεση αρχικών δοκιμών εγκυρότητας (baselines).

#### 1.2.2. Ενότητα Αναγνώρισης Ονοματισμένων Οντοτήτων (NER Toolbox)
Η ενότητα αυτή εστιάζει στην παραγωγή υψηλής ποιότητας επισημειωμένων δεδομένων για την εκπαίδευση μοντέλων Αναγνώρισης Ονοματισμένων Οντοτήτων (Named Entity Recognition).
*   **Semi-Automated Annotation Framework**: Πρόκειται για ένα εξειδικευμένο πλαίσιο ημιαυτόματης επισημείωσης που υιοθετεί την προσέγγιση "Human-in-the-loop" (HITL). Επιτρέπει στον χρήστη να επιβλέπει και να διορθώνει τις προβλέψεις του αλγορίθμου σε πραγματικό χρόνο μέσω μιας διαδραστικής διεπαφής σε Streamlit.

Η εργαλειοθήκη παραμένει δυναμική, με δυνατότητα ενσωμάτωσης πρόσθετων ενοτήτων (όπως εργαλεία Οπτικοποίησης και Αυτόματης Περίληψης) που θα υποστηρίξουν περαιτέρω τις ανάγκες του έργου σε επόμενα στάδια.

| Αποθετήριο (Φάκελος) | Ρόλος | Τι περιέχει |
| :--- | :--- | :--- |
| **Classification Toolbox** | Εργαλεία σχετικά με classification | Enriched Label Descriptions, Dataset Creation Scripts |
| **NER Toolbox** | Εργαλεία σχετικά με NER | Semi-Automated Annotation Framework |

## 1.3 Τεχνικές Απαιτήσεις & Εγκατάσταση

Η εργαλειοθήκη έχει αναπτυχθεί με γνώμονα τη φορητότητα και την ευκολία εγκατάστασης σε διαφορετικά λειτουργικά συστήματα (Windows, Linux, macOS). Για τη διασφάλιση της αναπαραγωγιμότητας των αποτελεσμάτων, ακολουθείται η τυπική ροή εργασίας του οικοσυστήματος της Python.

### 1.3.1. Περιβάλλον και Προαπαιτούμενα
*   **Γλώσσα Προγραμματισμού**: Απαιτείται η έκδοση Python 3.x και ο διαχειριστής πακέτων pip.
*   **Διαχείριση Περιβάλλοντος**: Συνιστάται η χρήση περιβάλλοντος (virtual environment - venv) για την αποφυγή συγκρούσεων μεταξύ των βιβλιοθηκών και τη διατήρηση της καθαρότητας του συστήματος.

### 1.3.2. Κύριες Εξαρτήσεις (Dependencies)
Η εργαλειοθήκη βασίζεται σε δοκιμασμένες βιβλιοθήκες ανοιχτού κώδικα για εργασίες Μηχανικής Μάθησης και NLP:
*   **Επεξεργασία Δεδομένων & ML**: `numpy`, `scipy`, `scikit-learn`, `tqdm`.
*   **Φυσική Επεξεργασία Γλώσσας**: `spaCy` (με χρήση του μοντέλου `el_core_news_sm` για τη βέλτιστη διαχείριση της ελληνικής γλώσσας).
*   **Διεπαφή Χρήστη**: `Streamlit` για την εκτέλεση της εφαρμογής ημιαυτόματης επισημείωσης (HITL).

### 1.3.3. Διαδικασία Εγκατάστασης
Η προετοιμασία της εργαλειοθήκης ολοκληρώνεται σε τρία βασικά βήματα:

1.  **Λήψη Κώδικα**: Κλωνοποίηση του κεντρικού αποθετηρίου από το GitHub:
    ```bash
    git clone https://github.com/syn-nomos/Syn-nomos-project.git
    ```
2.  **Δημιουργία Εικονικού Περιβάλλοντος**:
    ```bash
    python -m venv .venv
    # Ενεργοποίηση:
    # Windows: .venv\Scripts\activate
    # Linux/Mac: source .venv/bin/activate
    ```

### 1.3.4. Εξειδικευμένη Τεκμηρίωση ανά Υποσύστημα
Επισημαίνεται ότι, πέρα από τις γενικές οδηγίες, κάθε επιμέρους εργαλειοθήκη (π.χ. Classification, NER) συνοδεύεται από το δικό της αρχείο README.md στον αντίστοιχο φάκελο του αποθετηρίου. Εκεί περιλαμβάνονται:
*   Αναλυτικές οδηγίες χρήσης και παραμετροποίησης των scripts.
*   Εξειδικευμένες εξαρτήσεις (dependencies) που αφορούν αποκλειστικά το συγκεκριμένο task.
*   Παραδείγματα εκτέλεσης (Quickstart) για τη γρήγορη επιβεβαίωση της ορθής λειτουργίας του εργαλείου.
