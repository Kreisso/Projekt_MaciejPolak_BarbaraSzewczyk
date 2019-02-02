Projekt realizowany przez:
- Maciej Polak
- Barbara Szewczyk


Projekt ma na celu analizę danych 'Boston Housing DATA', oraz wykonanie modelu regresji liniowej nauczania maszynowego.

Stworzonę zostały dwa pliki wykonwacze:
- analiza.py
- main.py


Kod w pliku analiza.py służy do przeanalizowania danych jak i utworzenia modelu,
ponadto zostają utworzone pliki obrazujące:
- Macierz korelacji
- Rozkład zmiennej PRICE
- Wykres punktowy Price test vs Price predicted test
- Wykres punktowy Price train vs Price train test
- Wykres punktowy Price względem RM
- Wykres punktowy Price względem LSTAT
- Wykres punktowy Price względem PTRATIO
- Wykres punktowy Price względem INDUS
- Wykres punktowy Wykorzystania danych

Zostały utworzone trzy modele:
- Pierwszy uwzględnia tylko wpływ RM oraz LSTAT
- Drugi uwzględnia RM, LSTAT oraz PTRATIO
- Trzeci  uwzględnia RM, LSTAT, PTRATIO oraz INDUS

Dla każdego z tych modeli zostały utworzone wyżej wymienione pliki obrazujące
( Macierz korelacji jak i rozkład zmiennej prcie jest taki sam dla każdego modelu )


Kod w pliku main.py ma na celu załadowanie wszystkich utworzonych modeli oraz wyliczenie na ich podstawie:
- RMSE ( błąd średniowkadratowy )
- R2 ( współczynnik determinacji )