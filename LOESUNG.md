# Übungsblatt: Entscheidungsbäume

## Starke und schwache KI

Ein System wie ChatGPT zeigt in Gesprächen Verhalten, das intelligent wirken kann. Der Kern ist aber ein großes generatives Sprachmodell: Es verarbeitet den bisherigen Kontext und berechnet, welche Fortsetzung wahrscheinlich passt. Dazu kommen unter anderem Nachtraining, Sicherheitsregeln und Werkzeuge. Das System kann dadurch erklären, zusammenfassen und Probleme bearbeiten. Daraus folgt aber nicht automatisch, dass es wie ein Mensch versteht, eigene Absichten besitzt oder bewusst ist. Nach dieser Unterscheidung ist ChatGPT eine schwache beziehungsweise spezialisierte KI und keine nachgewiesene starke allgemeine KI.

Es gibt viele Systeme, die in einem begrenzten Bereich Leistungen erbringen, die man intelligent nennen kann. Beispiele sind Routenplanung, Bilderkennung, medizinische Entscheidungshilfen, Schachprogramme oder Roboter. Solche Systeme nehmen Daten auf und wählen anhand gelernter Modelle, Regeln oder Suchverfahren eine passende Ausgabe oder Handlung. Außerhalb ihres vorgesehenen Bereichs sind sie jedoch meist deutlich weniger zuverlässig.

Für viele Anwendungen reicht schwache KI aus. Entscheidend ist nicht, ob ein System wirklich denkt, sondern ob es seine Aufgabe zuverlässig, sicher und nachvollziehbar erfüllt. Gleichzeitig muss die Auswirkung betrachtet werden:

- **Vorteile:** Unterstützung bei Diagnose, Übersetzung, Planung, Barrierefreiheit und Automatisierung.
- **Unfälle:** Ein Modell kann wegen unpassender Trainingsdaten, falscher Korrelationen oder ungewohnter Eingaben Fehler machen.
- **Fairness und Bias:** Verzerrte Daten können bestimmte Personengruppen benachteiligen.
- **Sicherheit und Missbrauch:** Dieselben Fähigkeiten können auch für Spam, Betrug, Spear-Phishing oder Desinformation eingesetzt werden.
- **Dual Use:** Eine Technik kann nützliche und schädliche Zwecke haben, zum Beispiel bei Genbearbeitung, Kerntechnik oder Raketen.

Darum sollten Systeme nicht nach ihrer Absicht beurteilt werden, denn ein technisches System hat nicht zwingend eine eigene Absicht. Wichtiger sind die tatsächlichen und vorhersehbaren Auswirkungen, menschliche Kontrolle, Tests und klare Verantwortlichkeiten.

## DTL.01: ID3-Handsimulation

ID3 wählt in jedem Knoten das Attribut mit dem größten Informationsgewinn.

\[
H(S)=-\sum_i p_i\log_2(p_i)
\]

\[
R(S,A)=\sum_{v\in A}\frac{|S_v|}{|S|}H(S_v)
\qquad
Gain(S,A)=H(S)-R(S,A)
\]

In der gesamten Trainingsmenge kommen viermal Kandidat O und dreimal Kandidat M vor:

\[
H(S)=H(4/7,3/7)=0{,}9852
\]

### Wahl des Wurzelknotens

#### Alter

- `>= 35`: 2 O, 2 M, also \(H=1\)
- `< 35`: 2 O, 1 M, also \(H=0{,}9183\)

\[
R(S,Alter)=\frac47\cdot1+\frac37\cdot0{,}9183=0{,}9650
\]

\[
Gain(S,Alter)=0{,}9852-0{,}9650=0{,}0202
\]

#### Einkommen

- `hoch`: 3 O, 1 M, also \(H=0{,}8113\)
- `niedrig`: 1 O, 2 M, also \(H=0{,}9183\)

\[
R(S,Einkommen)=\frac47\cdot0{,}8113+\frac37\cdot0{,}9183=0{,}8571
\]

\[
Gain(S,Einkommen)=0{,}9852-0{,}8571=0{,}1281
\]

#### Bildung

- `Abitur`: 1 O, 2 M, also \(H=0{,}9183\)
- `Master`: 2 O, 0 M, also \(H=0\)
- `Bachelor`: 1 O, 1 M, also \(H=1\)

\[
R(S,Bildung)=\frac37\cdot0{,}9183+\frac27\cdot0+\frac27\cdot1=0{,}6793
\]

\[
Gain(S,Bildung)=0{,}9852-0{,}6793=0{,}3060
\]

`Bildung` besitzt den größten Informationsgewinn und wird deshalb die Wurzel.

### Weitere Knoten

Für `Master` sind beide Beispiele O. Dieser Ast ist sofort ein Blatt mit O.

Bei `Abitur` gibt es ein O und zwei M. Eine Aufteilung nach `Einkommen` trennt die Beispiele vollständig:

- Einkommen `hoch`: O
- Einkommen `niedrig`: M

Der Restwert ist 0 und der Informationsgewinn damit \(0{,}9183\). Eine Aufteilung nach Alter hätte nur einen Informationsgewinn von \(0{,}2516\).

Bei `Bachelor` gibt es ein O und ein M. `Alter` trennt beide Beispiele vollständig:

- Alter `>= 35`: M
- Alter `< 35`: O

Der Informationsgewinn beträgt hier 1. `Einkommen` kann die Beispiele nicht trennen, weil beide Einkommen `hoch` besitzen.

### Fertiger Baum

```text
Bildung?
├── Master       -> O
├── Abitur
│   └── Einkommen?
│       ├── hoch    -> O
│       └── niedrig -> M
└── Bachelor
    └── Alter?
        ├── >= 35 -> M
        └── < 35  -> O
```

Der Baum klassifiziert alle sieben Trainingsbeispiele richtig. Seine Trainingsfehlerrate ist daher 0 %.

## DTL.03: Machine Learning mit Weka

Verwendet wurde Weka 3.8.7. Für ID3 wurde über den Package Manager das Paket `simpleEducationalLearningSchemes` installiert. Die bearbeiteten Dateien liegen unter [`data`](data). Bei allen Auswertungen wurde der Trainingssatz zugleich als Testsatz verwendet, entsprechend der Option **Use training set**.

Der Tiername in `zoo` ist nur eine Kennung und wurde vor dem Training entfernt. Würde er als normales nominales Merkmal verwendet, könnte der Baum einzelne Tiere auswendig lernen.

### 1. J48 mit den CSV-Dateien

#### Zoo

Weka erkennt die 0/1-Werte und `legs` beim CSV-Import als numerisch. Der gelernte Baum lautet gekürzt:

```text
feathers <= 0
| milk <= 0
| | backbone <= 0
| | | airborne <= 0
| | | | predator <= 0
| | | | | legs <= 2: shellfish
| | | | | legs > 2: insect
| | | | predator > 0: shellfish
| | | airborne > 0: insect
| | backbone > 0
| | | fins <= 0
| | | | tail <= 0: amphibian
| | | | tail > 0: reptile
| | | fins > 0: fish
| milk > 0: mammal
feathers > 0: bird
```

- 100 von 101 Instanzen richtig
- 1 von 101 Instanzen falsch
- Trainingsfehlerrate: **0,9901 %**

```text
             vorhergesagt
             mam fis bir she ins amp rep
tatsächlich
mammal        41   0   0   0   0   0   0
fish           0  13   0   0   0   0   0
bird           0   0  20   0   0   0   0
shellfish      0   0   0  10   0   0   0
insect         0   0   0   0   8   0   0
amphibian      0   0   0   0   0   3   1
reptile        0   0   0   0   0   0   5
```

Die Diagonale enthält die richtigen Zuordnungen. Nur ein Amphibium wurde als Reptil klassifiziert.

#### Restaurant

```text
Patrons = Some: Yes
Patrons = Full: No
Patrons = None: No
```

- 10 von 12 Instanzen richtig
- 2 von 12 Instanzen falsch
- Trainingsfehlerrate: **16,6667 %**

```text
       vorhergesagt
       Yes No
Yes     4  2
No      0  6
```

Vier tatsächliche `Yes` und alle sechs tatsächlichen `No` wurden richtig erkannt. Zwei tatsächliche `Yes` wurden als `No` vorhergesagt. Die Standard-Pruning-Einstellung von J48 hält den Baum klein und nimmt dabei diese beiden Trainingsfehler in Kauf.

### 2. ARFF-Format und Attributtypen

Eine ARFF-Datei besteht aus einem Kopf mit `@relation` und `@attribute` sowie dem Datenbereich ab `@data`.

- **Nominal:** Eine feste Auswahl von Werten, zum Beispiel `{rot,grün,blau}`. Die Werte haben keine Reihenfolge.
- **Ordinal:** Kategorien mit natürlicher Reihenfolge, zum Beispiel `{klein,mittel,gross}`. ARFF besitzt keinen eigenen `ordinal`-Datentyp. Man kann die Reihenfolge als passend codierte numerische Werte darstellen, muss dann aber wissen, dass die Abstände nicht unbedingt inhaltlich gleich groß sind.
- **Numeric:** Zahlenwerte, mit denen Größenvergleiche und Schwellen möglich sind, zum Beispiel Alter oder Gewicht.
- **String:** Freier Text ohne vorher festgelegte Wertemenge, zum Beispiel ein Name oder Kommentar.

Die verwendete ID3-Implementierung verarbeitet nur nominale Eingabeattribute und keine numerischen oder String-Attribute. Deshalb wurden die binären Zoo-Merkmale und `legs` in [`zoo.arff`](data/zoo.arff) als nominale Wertemengen angegeben. `animal_name` bleibt als String dokumentiert, wird aber vor dem Training entfernt. Im Restaurantdatensatz sind alle Merkmale nominal.

### 3. ID3 und J48 mit den ARFF-Dateien

#### Zoo mit ID3

```text
legs = 0
| fins = 0
| | toothed = 0: shellfish
| | toothed = 1: reptile
| fins = 1
| | eggs = 0: mammal
| | eggs = 1: fish
legs = 2
| hair = 0: bird
| hair = 1: mammal
legs = 4
| hair = 0
| | aquatic = 0: reptile
| | aquatic = 1
| | | toothed = 0: shellfish
| | | toothed = 1: amphibian
| hair = 1: mammal
legs = 5: shellfish
legs = 6
| aquatic = 0: insect
| aquatic = 1: shellfish
legs = 8: shellfish
```

ID3 klassifiziert alle 101 Trainingsinstanzen richtig. Die Fehlerrate ist **0 %**. In der Konfusionsmatrix stehen deshalb nur Werte auf der Hauptdiagonalen: 41 Säugetiere, 20 Vögel, 5 Reptilien, 13 Fische, 4 Amphibien, 8 Insekten und 10 Schalentiere wurden richtig erkannt.

#### Zoo mit J48

J48 erreicht auch mit der ARFF-Datei 100 von 101 richtigen Zuordnungen und damit eine Fehlerrate von **0,9901 %**. Die Konfusionsmatrix entspricht inhaltlich dem CSV-Lauf: Ein Amphibium wird als Reptil eingeordnet. Weil `legs` in der ARFF-Datei nominal statt numerisch definiert ist, besitzt der Baum einzelne Äste für `legs = 0`, `legs = 2` usw. Der CSV-Baum verwendet dagegen numerische Schwellen wie `legs <= 2`. Der ARFF-Baum hat deshalb 13 Blätter statt 9, obwohl die Fehlerrate gleich bleibt.

#### Restaurant mit ID3

```text
Patrons = None: No
Patrons = Some: Yes
Patrons = Full
| Type = French: No
| Type = Thai
| | FriSat = Yes: Yes
| | FriSat = No: No
| Type = Burger
| | Alternate = Yes: Yes
| | Alternate = No: No
| Type = Italian: No
```

ID3 klassifiziert alle 12 Trainingsinstanzen richtig. Die Fehlerrate ist **0 %** und die Konfusionsmatrix lautet:

```text
       vorhergesagt
       Yes No
Yes     6  0
No      0  6
```

#### Restaurant mit J48

Der Baum und die Ergebnisse sind bis auf die Reihenfolge der Äste gleich wie beim CSV-Lauf:

```text
Patrons = None: No
Patrons = Some: Yes
Patrons = Full: No
```

Die Fehlerrate beträgt wieder **16,6667 %**; zwei `Yes`-Fälle werden als `No` klassifiziert.

### Vergleich

ID3 kürzt die Bäume nicht und erreicht deshalb auf diesen Trainingsdaten 0 % Fehler. J48 kürzt den Baum automatisch. Dadurch ist der Baum einfacher, macht beim Zoo-Datensatz aber einen Fehler und beim Restaurant-Datensatz zwei Fehler. Das heißt nicht automatisch, dass ID3 bei neuen Daten besser wäre. Dafür müsste man die Verfahren mit Daten testen, die beim Training noch nicht verwendet wurden. Auch der festgelegte Attributtyp beeinflusst den aufgebauten Baum.

## Quellen

- Aufgabenblatt: <https://github.com/Artificial-Intelligence-HSBI-TDU/KI-Vorlesung-W26/blob/master/homework/sheet-dtl.md>
- Weka und ARFF-Dokumentation: <https://waikato.github.io/weka-wiki/formats_and_processing/arff/>
- AIMA-Datensätze: <https://github.com/aimacode/aima-data>
- NIST AI Risk Management Framework: <https://doi.org/10.6028/NIST.AI.100-1>
