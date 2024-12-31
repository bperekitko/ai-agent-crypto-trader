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
