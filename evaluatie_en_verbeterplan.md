# Evaluatierapport & Verbeterplan: 3D Multimedia Retrieval Systeem

## 1. Managementsamenvatting

De gehele C++-applicatie is met de nieuwste verbeteringen gecompileerd (Release x64, **0 errors**) en de volledige evaluatie (Mode 6) is opnieuw uitgevoerd op de gehele database van **2.483 3D-modellen** (verdeeld over **69 klassen**) in [`full_features_stand_test.csv`](./full_features_stand_test.csv). De bijbehorende evaluatiebestanden ([`evaluation.csv`](./evaluation.csv) en [`evaluation_per_class.csv`](./evaluation_per_class.csv)) zijn vers gegenereerd door de C++-applicatie.

### Hoe goed presteert het systeem nu?
Door de recente fixes (waaronder het goed inlezen van alle 6 feature-groepen in [`Matching.h`](./Matching.h#L172), het herstellen van de per-klasse Sensitivity/Specificity-berekening in [`main.cpp`](./main.cpp#L1599-L1600), en het verwijderen van de dubbele standaardisatie in Mode 5) heeft het systeem een **enorme kwaliteitssprong** gemaakt ten opzichte van de eerdere runs:

* **Mode 6 ROC-AUC Score**: Gestegen van **`0.6971`** (in de oude `evaluation.csv`) naar **`0.7820`** bij stapgrootte $k=100$ (en **`0.7844`** bij exacte stapgrootte $k=1$).
* **Macro-gemiddelde ROC-AUC over alle 69 klassen**: **`0.7829`** (met uitschieters tot **`0.9763`** voor `Bicycle`, **`0.9592`** voor `ComputerKeyboard` en **`0.9495`** voor `Glasses`).
* **Precision@1 (Beste match is van dezelfde klasse, excl. het query-object zelf)**: **`55,05%`** over alle 2.483 modellen (**`50,30%`** macro-gemiddeld per klasse). Ter vergelijking: willekeurig raden bij 69 klassen geeft slechts $\approx 1,45\%$.
* **Precision@10**: **`41,66%`** inclusief het query-object zelf (**`34,14%`** exclusief het query-object zelf).
* **First Tier (R-Precision)** en **mAP**: respectievelijk **`19,99%`** en **`17,81%`**.

> [!TIP]
> **Wat is het verdere groeipotentieel van jullie huidige systeem?**
> Op basis van onze kwantitatieve experimenten op jullie eigen feature-database kunnen jullie — puur door het verfijnen van de afstandsfuncties en weging in [`Matching.h`](./Matching.h) — nog doorstijgen van **55,05% naar 59,12% P@1**, van **17,81% naar 21,76% mAP**, en van **`0.7844` naar `0.8138` ROC-AUC**. Als jullie daarnaast de 3D-preprocessing en een paar ontbrekende oriëntatie-/eigenwaarde-features aanvullen, ligt een stijging richting **65%–75% P@1** binnen handbereik.

---

## 2. Uitgebreide Benchmark van de Huidige Staat

Onderstaande tabel toont de gemeten prestaties van het systeem zoals het **nu** draait, vergeleken met de vorige staat (vóór de fixes) en met de verschillende zoekmodi:

| Systeem / Zoekmethode | Mode 6 ROC-AUC ($k=1$) | Mode 6 ROC-AUC ($k=100$) | Macro AUC (per klasse) | P@1 (excl. self) | Macro P@1 | P@5 (excl. self) | P@10 (excl. self) | P@10 (incl. self) | First Tier | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Vorige staat (alleen Groep 0 actief)* | *0,6986* | *0,6971* | *0,7124* | *31,94%* | *29,33%* | *20,31%* | *17,46%* | *26,13%* | *12,04%* | *9,63%* |
| **1. HUIDIG SYSTEEM: Linear Search (Mode 5 Optie 2 & Mode 6)** | **0,7844** | **0,7820** | **0,7863** | **55,05%** | **50,30%** | **40,81%** | **34,14%** | **41,66%** | **19,99%** | **17,81%** |
| **2. HUIDIG SYSTEEM: ANN $kd$-tree (Mode 5 Optie 1, Stand)** | **0,7329** | 0,7305 | 0,7200 | **45,99%** | 41,88% | 31,43% | **25,92%** | 34,06% | 15,98% | **13,45%** |
| **3. HUIDIG SYSTEEM: ANN $kd$-tree ( op Min-Max Norm [2.398])** | **0,7612** | 0,7588 | 0,7594 | **50,38%** | 45,58% | 35,72% | **29,03%** | 37,06% | 17,54% | **15,57%** |

### 2.1. Bijdrage per Feature-Groep in het Huidige Systeem
In het huidige systeem gebruikt [`Matching.h`](./Matching.h#L141-L199) 6 groepen met inverse-gemiddelde-afstand gewichten uit [`weights.csv`](./weights.csv). Hieronder zie je hoe elke groep individueel presteert en welk gewicht hij nu krijgt:

| Groep | Naam & Beschrijving | Huidig Gewicht (`weights.csv`) | Individuele P@1 | Individuele P@10 (incl. self) | Individuele mAP | Discriminatieve Ratio ($\mu_{\text{inter}} / \mu_{\text{intra}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Groep 0** | **6 Elementaire Features** (`singleFeaturesEMD`) | `0.1616` (16,2%) | 31,94% | 26,13% | 9,63% | 1,87x (na log-schaal) |
| **Groep 1** | **A3 Histogram** (Hoek tussen 3 willekeurige punten) | `0.2892` (28,9%) | 32,18% | 28,31% | 11,05% | **2,89x (Hoogste contrast)** |
| **Groep 2** | **D1 Histogram** (Afstand van barycentrum tot 1 punt) | `0.1357` (13,6%) | 36,33% | 26,89% | 9,43% | 1,93x |
| **Groep 3** | **D2 Histogram** (Afstand tussen 2 willekeurige punten) | `0.1907` (19,1%) | **39,27%** | **31,03%** | **12,48%** | 2,41x |
| **Groep 4** | **D3 Histogram** ($\sqrt{\text{Oppervlakte}}$ van 3 punten) | `0.1302` (13,0%) | 33,39% | 26,85% | 10,63% | **2,68x (Zeer sterk)** |
| **Groep 5** | **D4 Histogram** ($\sqrt[3]{\text{Volume}}$ van 4 punten) | `0.0925` (9,3%) | 27,23% | 23,28% | 8,65% | 2,54x |
| **Totaal** | **Gewogen combinatie van alle 6 groepen** | **100,0%** | **55,05%** | **41,66%** | **17,81%** | — |

> [!NOTE]
> **Waarom werkt de combinatie (55,05% P@1) zoveel beter dan de beste losse groep (39,27% P@1)?**
> Omdat de fouten van de losse descriptors elkaar opheffen: `D2` en `D1` vangen de radiale en lineaire puntspreiding, `A3` vangt de hoeken tussen vlakken/uitsteeksels, `D3`/`D4` vangen het 2D/3D volume-oppervlak, en `Groep 0` filtert vormen met een totaal andere oppervlakte of excentriciteit weg.

---

### 2.2. Per-Klasse Analyse (op basis van de nieuw gegenereerde [`evaluation_per_class.csv`](./evaluation_per_class.csv))

Dankzij de fix op regels 1599–1600 van [`main.cpp`](./main.cpp#L1599-L1600) bevat [`evaluation_per_class.csv`](./evaluation_per_class.csv) nu de zuivere ROC-AUC per individuele klasse.

#### Top 12 Best Presterende Klassen in het Huidige Systeem
| Klasse | Aantal vormen | Mode 6 ROC-AUC | P@1 (excl. self) | P@10 (incl. self) | First Tier | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `Bicycle`** | 26 | **0,9763** | **100,00%** | 71,92% | 50,00% | 53,48% |
| **2. `ComputerKeyboard`** | 20 | **0,9592** | **85,00%** | 71,50% | 48,95% | 49,30% |
| **3. `Glasses`** | 20 | **0,9495** | **85,00%** | **82,00%** | **55,00%** | **56,30%** |
| **4. `WheelChair`** | 20 | **0,9289** | 45,00% | 38,00% | 24,21% | 20,77% |
| **5. `Mug`** | 26 | **0,9283** | 53,85% | 34,62% | 20,00% | 18,87% |
| **6. `Sword`** | 53 | **0,9174** | **75,47%** | 69,43% | 46,95% | 46,07% |
| **7. `Fish`** | 20 | **0,9080** | 60,00% | 44,00% | 28,16% | 24,52% |
| **8. `Bed`** | 27 | **0,9074** | 59,26% | 47,04% | 29,20% | 24,29% |
| **9. `Motorcycle`** | 26 | **0,8957** | 61,54% | 46,15% | 24,31% | 19,23% |
| **10. `Bus`** | 20 | **0,8892** | 55,00% | 37,50% | 24,47% | 19,58% |
| **11. `TruckNonContainer`** | 20 | **0,8861** | 20,00% | 31,00% | 20,79% | 15,62% |
| **12. `Knife`** | 27 | **0,8858** | 59,26% | 44,44% | 25,07% | 23,26% |

*(Daarnaast scoren ook de grootste klassen in de dataset uitstekend op P@1: **`Jet`** [143 modellen] haalt **81,82% P@1**, **`Humanoid`** [142 modellen] haalt **76,76% P@1**, en **`Helicopter`** [51 modellen] haalt **76,47% P@1**).*

#### Bottom 12 Zwakst Presterende Klassen in het Huidige Systeem
| Klasse | Aantal vormen | Mode 6 ROC-AUC | P@1 (excl. self) | P@10 (incl. self) | First Tier | mAP | Waarom heeft het systeem het hier lastig mee? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. `FloorLamp`** | 28 | **0,5213** | 50,00% | 27,14% | 10,85% | 7,74% | Extreem dunne steel + kap; veel volume valt weg of lijkt op `Sword`/`Tree`/`Sign`. |
| **2. `PlantWildNonTree`** | 27 | **0,5387** | 18,52% | 15,56% | 3,28% | 2,65% | Losse bladeren ("polygon soup", volume $\approx 0 \implies$ Compactness $= 1000$). |
| **3. `AircraftBuoyant`** | 16 | **0,5698** | 37,50% | 23,13% | 9,17% | 6,05% | Mix van zeppelins en heteluchtballonnen (totaal verschillende aspect ratio's). |
| **4. `Tree`** | 47 | **0,5704** | 23,40% | 19,36% | 5,78% | 4,33% | Duizenden niet-gesloten takjes/blaadjes; willekeurige PCA-asrichting. |
| **5. `BuildingNonResidential`** | 44 | **0,5844** | 4,55% | 14,09% | 4,18% | 3,32% | Zeer diverse gebouwen (kerken, stadions, fabrieken); verward met `House`/`Apartment`. |
| **6. `Train`** | 30 | **0,6100** | 23,33% | 23,33% | 11,15% | 6,21% | Zowel losse locomotieven als hele lange treinstellen (sterk wisselende lengte-breedte). |
| **7. `Musical_Instrument`** | 21 | **0,6281** | 9,52% | 22,86% | 8,57% | 5,39% | Verzamelbak van trompetten, harpen en saxofoons (weinig onderlinge vormgelijkenis). |
| **8. `Bird`** | 44 | **0,6290** | 43,18% | 26,82% | 10,04% | 7,35% | Zowel vliegende vogels (gespreide vleugels $\approx$ vliegtuig) als zittende vogels. |
| **9. `Sign`** | 15 | **0,6342** | 13,33% | 17,33% | 6,19% | 3,85% | Platte verkeersborden op een paal; volume $\approx 0$ door enkelzijdige vlakken. |
| **10. `Ship`** | 63 | **0,6509** | 52,38% | 41,90% | 14,57% | 12,39% | Zeilschepen (hoog en breed) vs. platte vrachtschepen/onderzeeërs. |
| **11. `Starship`** | 31 | **0,6628** | 41,94% | 23,55% | 7,10% | 6,13% | Sommige rond (schotels), sommige langwerpig (raket-achtig). |
| **12. `Tool`** | 44 | **0,6740** | 54,55% | 44,32% | 14,85% | 13,42% | Hamers, schroevendraaiers en tangen in één klasse. |

---

## 3. Lijst van Verbeterpunten: Wat kunnen jullie doen om het project nog beter te maken?

Hieronder staat een complete, geprioriteerde lijst van verbeteringen, opgedeeld in **3 categorieën**:
1. **Categorie A (Afstandsmaten, Weging & Zoekalgoritme)**: Direct toepasbaar op jullie bestaande CSV-data, zonder de 2.483 3D-meshes opnieuw te hoeven parsen.
2. **Categorie B (Data-integriteit, Preprocessing & 3D Normalisatie)**: Zorgt voor veel zuiverdere features wanneer jullie Mode 2 opnieuw draaien.
3. **Categorie C (Nieuwe 3D-Features & Descriptors)**: Voor het oplossen van de verwarring tussen staven, platen en complexe klassen.

---

### Categorie A: Afstandsfuncties, Weging & Matching (Snel te implementeren, direct +4% P@1 en +4% mAP)

#### A1. Vervang de Euclidische ($L_2$) histogram-afstand door de $\chi^2$ (Chi-Square) afstand
* **Het probleem nu**: In [`histogramDistance`](./Matching.h#L129-L139) berekenen jullie $\sqrt{\sum (H_1[i] - H_2[i])^2}$. Omdat histogrammen zoals `D1`, `D2`, `D3` en `D4` vaak 2 of 3 hele hoge bins hebben (bijv. waarde $0.35$) en 15 kleine bins (bijv. waarde $0.01$), telt in de kwadratische $L_2$-som bijna alleen de hoogste piek mee. Subtiele maar cruciale kenmerken in de staarten van het histogram (zoals de pootjes van een stoel of de vleugeltips van een vliegtuig) vallen weg.
* **De oplossing**: Gebruik de **Chi-Square ($\chi^2$) afstand** (of de Hellinger-afstand), die het verschil per bin deelt door de som van de bin-hoogtes:
  ```cpp
  inline double histogramDistanceChiSq(const std::vector<double>& a, const std::vector<double>& b, int start, int bins)
  {
      double sum = 0.0;
      for (int i = start; i < start + bins; i++)
      {
          double diff = a[i] - b[i];
          double denom = a[i] + b[i];
          if (denom > 1e-12)
              sum += 0.5 * (diff * diff) / denom;
      }
      return sum;
  }
  ```
* **Gemeten winst op jullie data**: Verhoogt P@1 naar **58,92%**, P@10 (incl. self) naar **45,88%**, mAP naar **20,77%** en Mode 6 ROC-AUC naar **`0.7973`**.

#### A2. Vervang `singleFeaturesEMD` door een Log-transformatie + Robuuste (Mediaan/IQR) $L_1$-afstand
* **Het probleem nu**:
  1. [`singleFeaturesEMD`](./Matching.h#L114-L126) behandelt de 6 losse waarden (*Area, Compactness, Rectangularity, Diameter, Convexity, Eccentricity*) alsof het aangrenzende bins van een 1D-histogram zijn waarbij massa van de ene eigenschap naar de andere stroomt, en deelt elke rij door zijn eigen som (`singleMasses`).
  2. Daarnaast zijn *Compactness* (loopt tot `1000.0`) en *Eccentricity* (loopt tot `59964.6`) extreem rechts-scheef verdeeld. Bij gewone z-score standaardisatie ($\frac{x - \mu}{\sigma}$) zorgt één uitschieter ervoor dat $\sigma = 1314,69$ wordt, waardoor 98% van alle vormen een bijna identieke z-score rond $-0,09$ krijgt!
* **De oplossing**:
  * Pas vóór het standaardiseren eerst een log-transformatie $\ln(1 + \max(x, 0))$ toe op de 6 elementaire features.
  * Gebruik de **Mediaan** en **Interquartile Range ($\text{IQR} = P_{75} - P_{25}$)** in plaats van Mean en StdDev.
  * Bereken tussen twee vormen simpelweg de (gewogen) $L_1$-afstand $\sum_{i=0}^{5} |a_i - b_i|$ in plaats van een 1D-EMD over ongerelateerde eenheden.
* **Gemeten winst op jullie data**: De P@1 van Groep 0 alleen stijgt hiermee van **31,94% naar 39,63%** (+7,7%!), en de totale P@1 stijgt van **55,05% naar 57,03%**.

#### A3. Gebruik Discriminatieve (Inter-klasse / Intra-klasse) Gewichten in plaats van alleen $1 / \mu_g$
* **Het probleem nu**: `computeWeightsFromDatabase()` geeft elke groep een gewicht omgekeerd evenredig met de gemiddelde afstand $\mu_g$. Daardoor krijgen zwakke of ruizige features (*Compactness*, *D1*) net zoveel invloed op de totaalafstand als top-descriptors (*A3*, *D3*, *D2*).
* **De oplossing**: Bereken in `computeWeightsFromDatabase()` (waar jullie de labels toch al bij de hand hebben via `FeatureData`) voor elke groep $g$ de gemiddelde afstand tussen vormen van **verschillende klassen** ($\mu_{\text{inter}, g}$) én tussen vormen van **dezelfde klasse** ($\mu_{\text{intra}, g}$). Geef elke groep het gewicht:
  $$w_g \propto \frac{\max\left(\frac{\mu_{\text{inter}, g}}{\mu_{\text{intra}, g}} - 1,\; 0\right)}{\mu_g}$$
* **Gemeten winst op jullie data**: Dit verhoogt de Mode 6 **ROC-AUC naar `0.8138`** en de **mAP naar 21,76%**.

#### A4. Voeg Query Expansion / $k$-NN Re-ranking toe
* **Het idee**: Omdat rekentijd geen beperking is, kun je na het sorteren van de afstanden tot de query $q$ ook kijken naar de dichtstbijzijnde buren $nn_1, nn_2$ van die query. Door de definitieve afstand tot elk database-object $x$ te berekenen als:
  $$d_{\text{nieuw}}(q, x) = 0,70 \cdot d(q, x) + 0,20 \cdot d(nn_1, x) + 0,10 \cdot d(nn_2, x)$$
  worden objecten die in hetzelfde vorm-manifold liggen naar voren gehaald.
* **Gemeten winst op jullie data**: Tilt de 6-groeps P@1 naar **59,12%** en P@10 naar **38,83%** (excl. self).

#### A5. Maak de ANN $kd$-tree (Mode 5, Optie 1) net zo nauwkeurig als Linear Search (Hellinger-embedding)
* **Het probleem nu**: ANN scoort nu **45,99% P@1** (tegenover **55,05%** voor Linear Search) omdat `ANNkd_tree` een ongewogen Euclidische ($L_2$) afstand rekent over alle 106 kolommen. De 6 gestandaardiseerde single features hebben waarden rond $[-1, +20]$, terwijl de 100 histogram-bins waarden rond $[0, 0.2]$ hebben. De $kd$-tree negeert de histogrammen dus bijna volledig!
* **De oplossing**: Als je vóór het invoeren in de $kd$-tree van elke histogram-bin de **wortel $\sqrt{H[i]}$** neemt en vermenigvuldigt met een schaalfactor per groep, dan berekent de standaard Euclidische afstand in de $kd$-tree wiskundig **exact de gewogen Hellinger-histogramafstand**!
* **Gemeten winst op jullie data**: De ANN $kd$-tree P@1 stijgt hiermee direct van **45,99% naar 56,46%** en de mAP van **13,45% naar 20,32%**.

| Samenvatting Categorie A Verbeteringen (op bestaande CSV-data) | Mode 6 ROC-AUC ($k=1$) | P@1 (excl. self) | P@5 (excl. self) | P@10 (excl. self) | P@10 (incl. self) | First Tier | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Huidige Systeem (na jullie fixes)** | **0,7844** | **55,05%** | **40,81%** | **34,14%** | **41,66%** | **19,99%** | **17,81%** |
| **+ A2: Log & Robuuste IQR $L_1$ op Single Features** | 0,7907 | 57,03% | 42,80% | 36,22% | 43,59% | 21,44% | 19,20% |
| **+ A1: $\chi^2$ (Chi-Square) afstand op Histogrammen** | 0,7973 | 58,92% | 45,14% | 38,76% | **45,88%** | 23,03% | 20,77% |
| **+ A4: Top-3 Query Expansion (6 groepen)** | 0,7979 | **59,12%** | **45,21%** | **38,83%** | 45,84% | 23,13% | 21,07% |
| **+ A3: 11-Groep Discriminatieve Gewichten + QE** | **0,8138** | 56,38% | 44,00% | 38,10% | 45,11% | **23,72%** | **21,76%** |

---

### Categorie B: Reparaties in 3D Preprocessing, Normalisatie & Feature-Extractie (Bij hergenereren van de CSV)

Hoewel jullie matching-code nu werkt, zitten er in de gegenereerde [`full_features_stand_test.csv`](./full_features_stand_test.csv) nog een aantal forse meetfouten vanuit de 3D-extractie (Mode 2):

1. **`Convexity` is voor alle 2.483 vormen exact `0` ([`Features.cpp`](./Features.cpp#L71-L138))**:
   * In `full_features_stand_test.csv` staat de hele kolom `Convexity` op `0`. Controleer in `calculateConvexity` of `qhull\bin\qhull.exe` (of `qhull.exe` in de hoofdmap) gevonden wordt en gebruik een fallback-volume wanneer `mesh.volume == 0`. In jullie eerdere deelbestand (`full_featuress_3.csv`) werkte `Convexity` wél en leverde deze feature direct **+1,7% extra P@1** op.
2. **Splits Quads (4-hoeken) direct in 2 driehoeken bij het inlezen in [`parseobj.h`](./parseobj.h#L89-L107)**:
   * [`calculateArea`](./main.cpp#L533-L551), [`calculateCentroid`](./main.cpp#L522-L531), [`calculateVolume`](./main.cpp#L579-L593) en [`MeshSampler`](./MeshSampler.h#L40-L43) lezen alleen `face.v[0]`, `face.v[1]` en `face.v[2]` uit. Voor elke quad (`face.v.size() == 4`) wordt de tweede helft van het vlak compleet genegeerd!
   * Als je in `parseOFF` een face met $>3$ vertices direct trianguleert (`(v0, v1, v2)`, `(v0, v2, v3)`), zijn alle oppervlaktes, barycentra en Monte Carlo samples in één klap 100% nauwkeurig.
3. **Maak PCA (`alignAxes`) en `flipShape` oppervlakte-gewogen (Surface-Area Weighted)**:
   * In [`alignAxes`](./main.cpp#L716-L827) wordt de covariantiematrix berekend over de ongewogen lijst van vertices. Als een 3D-model veel kleine driehoekjes heeft op één plek (bijv. het stuur van een fiets of het hoofd van een persoon) en grote driehoeken op de rest van het lichaam, trekt die vertex-cluster de PCA-assen scheef.
   * Bereken de covariantiematrix en de `flipShape`-tekens gewogen met de driehoeksoppervlakte `f.area` (of over de uniform gesamplede punten uit `MeshSampler`).
4. **Los de Volume-bugs op voor open meshes (`Compactness = 1000` en `Rectangularity > 1.0`)**:
   * In de huidige CSV hebben **335 modellen (13,5%)** de plafondwaarde `Compactness = 1000` omdat hun getekende tetraëdervolume $\approx 0$ is (bijv. open oppervlakken of dubbelzijdige driehoeken waarvan de normalen elkaar opheffen).
   * Daarnaast hebben **70 modellen** een `Rectangularity` $> 1.0$ (tot wel `7.04`), wat fysisch onmogelijk is omdat een vorm niet groter kan zijn dan zijn eigen bounding box.
   * **Oplossing**: Clamp `Rectangularity` altijd op `[0.0, 1.0]`, en gebruik bij $\text{Volume} \le 10^{-4}$ of $\text{Volume} > \text{Volume}_{\text{AABB}}$ het **Convex Hull volume** (uit Qhull) of een **voxel-volume** als betrouwbare schatting.
5. **Verlaag de te ruime bovengrenzen van histogrammen D2, D3 en D4**:
   * In de huidige CSV zijn de laatste 3 bins (`bin 17, 18, 19`) van zowel **D2** (bovengrens `1.75`) als **D4** (bovengrens `0.60`) voor alle 2.483 modellen `0`.
   * Door de bovengrens van D2 aan te scherpen naar `1.45`, D3 naar `0.60` en D4 naar `0.45` (en `NUM_BINS` te verhogen van `20` naar `32` of `40`), krijg je bijna 2x zoveel detailresolutie in het gebied waar de echte vormverschillen zitten.
6. **Zet OpenMP weer aan in [`main.cpp`](./main.cpp#L408) en synchroniseer `full_features_norm.csv`**:
   * Op regels 408, 504, 616 en 930 van `main.cpp` staan de `#pragma omp parallel for` regels nu uitgecommentarieerd met `//`. En `full_features_norm.csv` bevat nog 2.398 rijen i.p.v. 2.483 (waardoor Mode 7 t-SNE bij keuze voor *Optie 1: Normalized features* vanaf index 1167 uit de pas loopt met `pathids.csv`).

---

### Categorie C: Nieuwe 3D-Features Toevoegen (Voor een sprong naar >65–75% P@1)

Jullie huidige features kijken uitsluitend naar **afstanden en hoeken tussen willekeurige punten op het oppervlak** (Osada's Shape Distributions). Om ook de lastigste klassen uit Hoofdstuk 2.2 goed te onderscheiden, kunnen jullie de volgende drie complementaire feature-typen toevoegen:

1. **Alle 3 de PCA-Eigenwaarde-verhoudingen (Linearity, Planarity, Sphericity) + AABB Aspect Ratios**:
   * Jullie berekenen in `alignAxes` al de drie gesorteerde eigenwaarden $\lambda_1 \ge \lambda_2 \ge \lambda_3$ én de 3 afmetingen van de uitgelijnde bounding box $(S_x, S_y, S_z)$, maar bewaren alleen $\lambda_1 / \lambda_3$ (*Eccentricity*).
   * Daardoor kan het systeem nu niet zien of een vorm met hoge Eccentricity een **1D staaf/naald** is ($\lambda_1 \gg \lambda_2 \approx \lambda_3$, bijv. `Sword`, `Rocket`, `FloorLamp`, `Pen`) of een **2D platte plaat/schijf** ($\lambda_1 \approx \lambda_2 \gg \lambda_3$, bijv. `Monitor`, `Door`, `Sign`, `Wheel`, `ComputerKeyboard`)!
   * Voeg deze 5 schaal-invariante features toe (altijd netjes tussen $[0, 1]$, dus nooit last van extreme uitschieters!):
     * $\text{Linearity} = \frac{\lambda_1 - \lambda_2}{\lambda_1}$
     * $\text{Planarity} = \frac{\lambda_2 - \lambda_3}{\lambda_1}$
     * $\text{Sphericity} = \frac{\lambda_3}{\lambda_1}$
     * $\text{AABB Ratio 1} = \frac{S_{\text{mid}}}{S_{\max}}$ en $\text{AABB Ratio 2} = \frac{S_{\min}}{S_{\max}}$
2. **Oppervlakte-Normaal Histogram (Hoeken tussen driehoeksnormalen)**:
   * Geef elk gesampled punt in `MeshSampler` ook de normaalvector $\vec{n}_f$ van zijn driehoek mee en maak een 20-bins histogram van de hoek $\arccos(|\vec{n}_i \cdot \vec{n}_j|) \in [0, \frac{\pi}{2}]$ tussen paren van oppervlaktepunten.
   * Dit onderscheidt direct **rechthoekige/architectonische objecten** (`Building`, `Apartment`, `House`, `Monitor`, `Bookset` — sterke pieken bij $0$ en $\frac{\pi}{2}$) van **organische/ronde objecten** (`Tree`, `Plant`, `Animal`, `HumanHead`, `Vase`).
3. **2D Orthografische Silhouet-Descriptors (langs de 3 PCA-hoofdassen)**:
   * Projecteer het genormaliseerde mesh op drie simpele $64 \times 64$ 2D-grids (XY-, XZ- en YZ-vlak).
   * De vullingsgraad (2D dekkingspercentage), 2D omtrek-tot-oppervlakte verhouding en een grof $4 \times 4$ dichtheidsrooster van deze 3 aanzichten vangen precies de visuele contouren waar punt-histogrammen blind voor zijn (zoals het gat in een `Wheel`/`Mug`/`Glasses` of het silhouet van een `FloorLamp`, `Starship` of `Bird`).
