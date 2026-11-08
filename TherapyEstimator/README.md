# Therapy estimator (prototyp)

Skattar basal, insulinkänslighet (ISF) och kolhydratkvot (CR) per tidsblock på dygnet från Loop-data
i Nightscout. Verktyget **föreslår** inställningar. Det ändrar ingenting i Loop.

## Kör

```sh
cd TherapyEstimator
export NS_URL=https://din-nightscout.example.com
export NS_TOKEN=...            # token med rollen "readable" (Admin Tools i Nightscout)
python3 -m therapy_estimator fetch --days 60          # sparar till data/
python3 -m therapy_estimator estimate --tune          # rapport + förslag
python3 -m therapy_estimator check                    # modellfria kontroller (korrigeringar, nätter, känningar)
python3 -m therapy_estimator podage                   # insulineffekt per poddag jämfört med dag 1
```

Flaggor för `estimate`: `--insulin {rapid-adult,rapid-child,fiasp,lyumjev}` (välj samma som i Loop),
`--blocks 0,4,8,12,16,20` (blockens starttimmar), `--json fil.json`, `--meal-groups 4,10,15,21`
(egen absorptionskurva per måltid; bra för att se absorptionstider, men kan dra ner ISF i simuleringar).

Testa med syntetisk data: `python3 -m therapy_estimator synth --out data_synth` och
`python3 -m therapy_estimator estimate --data data_synth`. Tester: `python3 -m pytest tests`.

## Hur det fungerar

Samma modell som Loop: glukosförändringen i varje 5-minuterssteg är insulinverkan × ISF, plus
leverns glukosproduktion (ISF × "sann" basal), plus absorberade kolhydrater × ISF/CR. Insulinkurvan är
Loops exponentiella modell. Schemalagd basal (som Loop inte laddar upp) räknas fram ur profilen.

Modellen är linjär i parametrarna och anpassas med robust minsta kvadrat-metod över 30-minutersfönster:

- dina nuvarande inställningar är utgångsläge (prior), och grannblock hålls nära varandra,
- formen på kolhydratabsorptionen lärs från datan (annars smittar fel i formen av sig på ISF),
- en långsam drift per halvdygn fångar sådant som sjukdom, så att det inte tolkas som ändrad ISF,
- känningar (+2 h), aktiva overrides, sensorstart, podbyte och orimliga hopp i CGM filtreras bort.

Validering: tränar på de första 75 % av dagarna och jämför prediktionsfelet mot dina nuvarande
inställningar på de sista 25 %. Lita bara på förslagen om "skattad" är tydligt bättre än "nuvarande".

## Begränsningar

- Förslagen begränsas till ±20 % per körning. Blocket hoppas över om det är för osäkert eller har för få måltider.
- 80 %-intervallen kommer från bootstrap över dagar. De fångar inte systematiska modellfel, så i
  simuleringar innehåller de sanningen oftare 60 % än 80 % av gångerna.
- Insulintoppen går dåligt att skatta, eftersom skillnaderna i `--tune` ofta är små. Använd Loops val om den är osäker.
- Kolhydrater som inte loggas (t.ex. druvsocker vid känning) ger fel. Logga dem om du kan.
- Loops automatiska bolusar reagerar på störningar modellen inte ser (t.ex. fett och protein), vilket
  får insulinet att se svagare ut. I simuleringar ger det upp till cirka 30 % för låg ISF. Att använda
  manuella bolusar som instrument (2SLS) prövades och gav sämre resultat, eftersom de följer måltiderna.
- På riktig data förklarar modellen mindre än på syntetisk. Jämför alltid med `check`, och ändra några
  få saker i taget.
- På syntetisk data med kända värden hamnar ISF inom ±20 %, basal inom ±15 % och
  kolhydratkänsligheten inom ±20 %.
