### 0. **Accuracy:**
   - W przypadku problemu z trzema klasami (gdzie losowe zgadywanie daje około 33%), warto uznać, że dokładność powyżej 50-60% wskazuje na lepszą niż losowa skuteczność. Lepsze klasyfikatory często osiągają 70% lub więcej, chociaż wartość ta może być różna w zależności od trudności zadania.


### 1. **Log Loss (Logarytmiczna Funkcja Straty):**

- **Co mierzy?**
  Log Loss, nazywane również Cross-Entropy Loss, mierzy niepewność przewidywań modelu. Uwzględnia zarówno prawidłowość
  klasyfikacji, jak i przewidywane prawdopodobieństwa.

- **Jak się interpretuje?**
    - Niższe wartości są lepsze. Idealny model, który przewiduje z doskonałą pewnością, ma log loss równy 0.
    - Wyższa wartość wskazuje na złe dopasowanie przewidywanych prawdopodobieństw do rzeczywistych wyników.
    - Idealna wartość to 0, ale wartości bliskie 0.69 mogą być uznawane za dobre w zależności od problemu, co symbolizuje redukcję niepewności w porównaniu do przypadku całkowicie losowych prognoz. Cokolwiek znacznie powyżej 1 wskazuje na problem z kalibracją prawdopodobieństw.
### 2. **ROC AUC (Receiver Operating Characteristic - Area Under Curve):**

- **Co mierzy?**
    - ROC AUC mierzy zdolność modelu do rozróżniania pomiędzy klasami. Pokazuje, jak dobrze model radzi sobie z różnymi
      progami klasyfikacji.

- **Jak się interpretuje?**
    - AUC równy 0.5 wskazuje na model przewidujący na poziomie losowym, a zbliżający się do 1 wskazuje na świetną
      zdolność rozpoznawania.
    - AUC poniżej 0.5 oznacza, że model działa gorzej niż losowy wybór.
    - Dla wieloklasowego problemu, średnia ROC AUC powinna przekraczać 0.7, aby uznać model za akceptowalnie skuteczny. Wartości powyżej 0.8 są często uważane za dobre, a powyżej 0.9 za bardzo dobre.

### 3. **Brier Score:**
**Jak działa:**

- **Wzór (dla problemu wieloklasowego):**
  $$
  \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (f_{ik} - o_{ik})^2
  $$
  - **N** – liczba obserwacji.
  - **K** – liczba klas.
  - $$f_{ik} \text { – przewidywane prawdopodobieństwo dla obserwacji } (i) \text{ i klasy } (k)$$
  - $$o_{ik} \text { – rzeczywisty wynik (1, jeśli obserwacja } ( i ) \text{ należy do klasy } ( k ), \text{0 w przeciwnym razie}$$
  
- **Co mierzy?**
    - Brier Score kwantyfikuje różnicę między przewidywanymi prawdopodobieństwami a rzeczywistymi wynikami (binarnymi).
      To miara średniego błędu kwadratowego.

- **Jak się interpretuje?**
    - Skala wynosi od 0 do 1. Niższe wartości wskazują na bardziej wiarygodne predykcje probabilistyczne.
    - Wskazuje, jak dobrze model przewiduje rozkład wyników dla każdej klasy.
    - Celem jest jak najniższa wartość. Wartości poniżej 0.2 mogą być uznane za dobre w wielu przypadkach. Pamiętaj, że interpretacja może się różnić w zależności od specyfiki i skali problemu.

### 4. **Kappa Cohena:**

- **Co mierzy?**
    - Kappa Cohena mierzy poziom zgodności pomiędzy dwoma klasyfikatorami, uprzedzając przypadkową zgodność.
    - Wartości powyżej 0.2 wskazują na nieco lepszą zgodność niż przypadkowa. Wartości powyżej 0.4-0.6 są zazwyczaj uważane za umiarkowaną zgodność, a powyżej 0.6 za dobrą zgodność.

- **Jak się interpretuje?**
    - Skala od -1 do 1. Wartość 1 reprezentuje doskonałą zgodność, 0 przedstawia zgodność na poziomie przypadku, a
      wartości ujemne oznaczają niezgodność większą niż przypadek.
### 5. **Precyzja (Precision)**

**Opis:**

- **Precyzja** mierzy, **jak dokładne są pozytywne przewidywania modelu**.
- Odpowiada na pytanie: **Spośród wszystkich przypadków, które model przewidział jako pozytywne (np. 'up'), ile z nich faktycznie jest pozytywnych?**
- Skupia się na **jakości przewidywań pozytywnych**.
- Wysoka precyzja oznacza, że **gdy model przewiduje pozytywny wynik, jest duża szansa, że ma rację**.
- **Minimalizuje liczbę fałszywych alarmów** (fałszywych pozytywów).

**Wzór:**
$$
\text{Precyzja} = \frac{\text{Prawdziwe Pozytywne}}{\text{Prawdziwe Pozytywne} + \text{Fałszywe Pozytywne}}
$$


### 6. **Czułość (Recall, Sensitivity)**

**Opis:**

- **Czułość** mierzy, **jak dobrze model wykrywa wszystkie pozytywne przypadki**.
- Odpowiada na pytanie: **Spośród wszystkich rzeczywistych pozytywnych przypadków (np. rzeczywistych 'up'), ile z nich model poprawnie zidentyfikował?**
- Skupia się na **zdolności modelu do wykrywania wszystkich pozytywnych przypadków**.
- Wysoka czułość oznacza, że **model wykrywa większość (lub wszystkie) rzeczywiste pozytywne przypadki**.
- **Minimalizuje liczbę przeoczonych przypadków** (fałszywych negatywów).

**Wzór:**

$$
\text{Czułość} = \frac{\text{Prawdziwe Pozytywne}}{\text{Prawdziwe Pozytywne} + \text{Fałszywe Negatywne}}
$$

### 7. F1-score
**Opis:**
F1-score to średnia harmoniczna precyzji i czułości. Jest używana, gdy chcemy znaleźć równowagę między precyzją a czułością, zwłaszcza przy niezbalansowanych danych.

**Jak działa:**
- **Wzór:** F1 = 2 * (Precyzja * Czułość) / (Precyzja + Czułość)
- **Dla każdej klasy** obliczasz F1-score osobno.

**Przykład dla klasy 'neutral':**
- **Precyzja:** Załóżmy 80%
- **Czułość:** Załóżmy 70%
- **F1-score:** 2 * (0,8 * 0,7) / (0,8 + 0,7) ≈ 74%

**Akceptowalne i bardzo dobre wartości:**
- **Akceptowalne:** F1-score powyżej 70%
- **Bardzo dobre:** F1-score powyżej 80%

### 8. Matthewsa Współczynnik Korelacji (MCC)

**Opis:**

- **MCC** to metryka, która mierzy jakość klasyfikacji, biorąc pod uwagę zarówno prawdziwe, jak i fałszywe pozytywy oraz negatywy.
- Jest uważana za **zrównoważony wskaźnik**, nawet przy niezbalansowanych danych.
- **W przypadku klasyfikacji wieloklasowej**, MCC może być uogólniony lub obliczony jako średnia ważona współczynników MCC dla każdej klasy.
- Wartość **MCC** wynosi od **-1** do **+1**:
    - **+1** oznacza idealną klasyfikację,
    - **0** losowe przewidywania,
    - **-1** całkowicie błędną klasyfikację.
    - **Akceptowalne wartości:** > 0,5
    - **Bardzo dobre wartości:** > 0,7
  
### 9. Krzywa Kalibracji i Błąd Kalibracji (Calibration Curve & Calibration Error)
**Opis:**

- **Krzywa Kalibracji** pokazuje, jak dobrze przewidywane prawdopodobieństwa modelu są skalibrowane w stosunku do rzeczywistych wyników.
- **Błąd Kalibracji** mierzy różnicę między przewidywanymi prawdopodobieństwami a rzeczywistymi częstościami występowania klas.
- **Błąd Oczekiwany (Expected Calibration Error, ECE):**
  $$
  \text{ECE} = \sum_{m=1}^{M} \frac{n_m}{N} \left| \text{przew}_m - \text{rzeczyw}_m \right|
  $$
  - **\( M \)**: liczba binów
  - **\( n_m \)**: liczba próbek w binie m
  - **\( N \)**: całkowita liczba próbek
  - **przew_m**: średnie przewidywane prawdopodobieństwo w binie m
  - **rzeczyw_m**: rzeczywista częstość pozytywnej klasy w binie m

**Akceptowalne i bardzo dobre wartości:**

- **Akceptowalne:** ECE poniżej **0,1** (10%)
- **Bardzo dobre:** ECE poniżej **0,05** (5%)