### 1. **Log Loss (Logarytmiczna Funkcja Straty):**

- **Co mierzy?**
  Log Loss, nazywane również Cross-Entropy Loss, mierzy niepewność przewidywań modelu. Uwzględnia zarówno prawidłowość
  klasyfikacji, jak i przewidywane prawdopodobieństwa.

- **Jak się interpretuje?**
    - Niższe wartości są lepsze. Idealny model, który przewiduje z doskonałą pewnością, ma log loss równy 0.
    - Wyższa wartość wskazuje na złe dopasowanie przewidywanych prawdopodobieństw do rzeczywistych wyników.

### 2. **ROC AUC (Receiver Operating Characteristic - Area Under Curve):**

- **Co mierzy?**
    - ROC AUC mierzy zdolność modelu do rozróżniania pomiędzy klasami. Pokazuje, jak dobrze model radzi sobie z różnymi
      progami klasyfikacji.

- **Jak się interpretuje?**
    - AUC równy 0.5 wskazuje na model przewidujący na poziomie losowym, a zbliżający się do 1 wskazuje na świetną
      zdolność rozpoznawania.
    - AUC poniżej 0.5 oznacza, że model działa gorzej niż losowy wybór.

### 3. **Brier Score:**

- **Co mierzy?**
    - Brier Score kwantyfikuje różnicę między przewidywanymi prawdopodobieństwami a rzeczywistymi wynikami (binarnymi).
      To miara średniego błędu kwadratowego.

- **Jak się interpretuje?**
    - Skala wynosi od 0 do 1. Niższe wartości wskazują na bardziej wiarygodne predykcje probabilistyczne.
    - Wskazuje, jak dobrze model przewiduje rozkład wyników dla każdej klasy.

### 4. **Kappa Cohena:**

- **Co mierzy?**
    - Kappa Cohena mierzy poziom zgodności pomiędzy dwoma klasyfikatorami, uprzedzając przypadkową zgodność.

- **Jak się interpretuje?**
    - Skala od -1 do 1. Wartość 1 reprezentuje doskonałą zgodność, 0 przedstawia zgodność na poziomie przypadku, a
      wartości ujemne oznaczają niezgodność większą niż przypadek.