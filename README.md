# Temat 16: Analiza i wizualizacja wybranych neuronów/obwodów w małym modelu sieci neuronowej
[![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml) 
[![Pytest](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml) 
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>

### Autorzy: 

## Opis projektu 
Celem projektu jest realizacja metody pozwalającej na zinterpretowanie procesu podejmowania decyzji przez model (dowolną, małą sieć neuronową).
Wyniki metody przedstawiane są przez wizualizację struktury sieci. W przeprowadzonym eksperymencie sprawdzamy, czy uzyskane w ten sposób informacje
mogą pozwolić na usprawnienie procesu optymalizacji lub regularyzacji.

## Wytyczne
**Wymagania:**
- [ ] Narzędzie, które dla (dowolnej) sieci neuronowej wizualizuje jej strukturę z naciskiem na interpretowalność decyzji modelu
- [ ] Wykorzystanie narzędzia w celu lepszej optymalizacji i regularyzacji modelu - eksperyment.
- [ ] Wyjaśnienie procesu podejmowania decyzji przez model
      
**Na plus:**
- [ ] Analiza wewnętrznych aktywacji: zidentyfikowanie neuronów/warstw, które mogą reagować na niepożądane wzorce (np. stereotypy, błędy).
- [ ] Eksperyment interwencyjny: manipulacja aktywacjami/cechami i obserwacja wpływu na wynik modelu.
- [ ] Wnioski związane z bezpieczeństwem: czy można wykryć i „wyłączyć” niebezpieczne wzorce? Jak architektura/model można zmodyfikować, by zwiększyć interpretowalność i bezpieczeństwo?

**Materiały:**
- [praca naukowa](https://arxiv.org/pdf/2501.16496)
- [mechanisticinterpretability github](https://github.com/apartresearch/mechanisticinterpretability)
- [artykuł](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms)

Więcej materiałów:
- [mechanistic interpretability with sparse autoencoder (youtube)](https://www.youtube.com/watch?v=UGO_Ehywuxc) - trochę intuicji stojącej za metodą korzystającą z SAE na przykładzie llmów
